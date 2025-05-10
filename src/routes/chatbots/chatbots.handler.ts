import { eq } from "drizzle-orm";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";
import { v4 as uuidv4 } from "uuid";

import type { AppRouteHandler } from "@/lib/types";

import db from "@/db";
import { loadPDFIntoPinecone } from "@/db/pinecone";
import { chatbots } from "@/db/schema";
import env from "@/env";
import { ZOD_ERROR_CODES, ZOD_ERROR_MESSAGES } from "@/lib/constants";
import { TrainingError } from "@/lib/error";
import { sendTrainingRequestWithoutRetry } from "@/lib/send-training-request-without-retry";

import type { CreateChatbot, DeleteChatbot, ListChatbot, PatchChatbot } from "./chatbots.routes";

// Get the current directory equivalent to __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const list: AppRouteHandler<ListChatbot> = async (c) => {
  const userId = c.get("userId");

  if (!userId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }

  const chatbots = await db.query.chatbots.findMany({
    where(fields, operators) {
      return operators.eq(fields.userId, userId);
    },
  });

  return c.json(chatbots, HttpStatusCodes.OK);
};

export const create: AppRouteHandler<CreateChatbot> = async (c) => {
  const userId = c.get("userId");

  if (!userId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }

  const formData = c.req.valid("form");
  const { pdf, ...chatbotData } = formData;

  if (!pdf || !(pdf instanceof File)) {
    return c.json(
      { message: "PDF file is required" },
      HttpStatusCodes.BAD_REQUEST,
    );
  }

  // validate pdf size max 10MB
  if (pdf.size > 10 * 1024 * 1024) {
    return c.json(
      { message: "PDF file size exceeds 10MB" },
      HttpStatusCodes.BAD_REQUEST,
    );
  }

  const [newChatbot] = await db.insert(chatbots).values({
    title: chatbotData.title,
    description: chatbotData.description,
    commandTemplate: chatbotData.commandTemplate,
    modelAi: chatbotData.modelAi,
    embedingModel: chatbotData.embedingModel,
    sugestionMessage: chatbotData.sugestionMessage,
    pdfTitle: pdf.name,
    pdfLink: pdf.name,
    userId: Number(userId),
  }).returning({
    id: chatbots.id,
    title: chatbots.title,
    modelAi: chatbots.modelAi,
  });

  try {
    // Simpan data chatbot ke database

    // Proses PDF ke Pinecone
    const pdfBuffer = await pdf.arrayBuffer();
    const pdfBlob = new Blob([pdfBuffer], { type: pdf.type });

    if (chatbotData.embedingModel === "pinecone") {
    // 1. Upload ke Pinecone
      await loadPDFIntoPinecone({
        pdfBlob,
        namespace: newChatbot.id.toString(),
      });
    }
    else {
      const form = new FormData();

      // Tambahkan file PDF
      form.append("pdf", pdfBlob, "document.pdf");

      // Tambahkan metadata
      form.append("userId", String(userId));
      form.append("chatbotId", String(newChatbot.id));
      form.append("modelType", chatbotData.embedingModel);
      form.append("pdfTitle", pdf.name);
      console.log("======Form Data:", form);

      // Kirim request ke Python server
      // Kirim request training dengan retry
      const trainingResponse = await sendTrainingRequestWithoutRetry(
        form,
        env.API_PASSWORD,
      );
      console.log("======Training Response:", trainingResponse);

      if (!trainingResponse.ok) {
        const errorBody = await trainingResponse.json();
        throw new TrainingError("Training failed", errorBody);
      }
    }

    return c.json({ message: "Chatbot created successfully" }, HttpStatusCodes.CREATED);
  }
  catch (error) {
    console.error("Error processing request:", error);

    // Rollback: Hapus chatbot jika gagal
    await db.delete(chatbots).where(eq(chatbots.id, newChatbot.id));

    return c.json(
      {
        message: HttpStatusPhrases.INTERNAL_SERVER_ERROR,
        details: error instanceof Error ? error.message : "Unknown error",
      },
      HttpStatusCodes.INTERNAL_SERVER_ERROR,
    );
  }
};

export const patch: AppRouteHandler<PatchChatbot> = async (c) => {
  const userId = c.get("userId");

  if (!userId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }

  const { id } = c.req.valid("param");

  const updates = c.req.valid("json");

  if (Object.keys(updates).length === 0) {
    return c.json(
      {
        success: false,
        error: {
          issues: [
            {
              code: ZOD_ERROR_CODES.INVALID_UPDATES,
              path: [],
              message: ZOD_ERROR_MESSAGES.NO_UPDATES,
            },
          ],
          name: "ZodError",
        },
      },
      HttpStatusCodes.UNPROCESSABLE_ENTITY,
    );
  }
  const [chatbot] = await db.update(chatbots)
    .set(updates)
    .where(eq(chatbots.id, id))
    .returning();

  if (!chatbot) {
    return c.json(
      {
        message: HttpStatusPhrases.NOT_FOUND,
      },
      HttpStatusCodes.NOT_FOUND,
    );
  }
  return c.json({
    message: "Chatbot updated successfully",
  }, HttpStatusCodes.OK);
};

export const remove: AppRouteHandler<DeleteChatbot> = async (c) => {
  const userId = c.get("userId");

  if (!userId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }

  const { id } = c.req.valid("param");

  // Check if the chatbot exists
  const chatbot = await db.query.chatbots.findFirst({
    where(fields, operators) {
      return operators.and(
        operators.eq(fields.id, id),
        operators.eq(fields.userId, userId),
      );
    },
  });
  if (!chatbot) {
    return c.json(
      {
        message: HttpStatusPhrases.NOT_FOUND,
      },
      HttpStatusCodes.NOT_FOUND,
    );
  }

  let successRemoveModelandStorage: boolean = false;

  // Check embedding model
  if (chatbot?.embedingModel === "pinecone") {
    successRemoveModelandStorage = true;
  }
  else {
    // Remove from ./storage/{userId}/{chatbotId}
    const storagePath = path.resolve(__dirname, "../../../storage");
    const chatbotPath = path.join(storagePath, userId.toString(), chatbot.id.toString());

    if (fs.existsSync(chatbotPath)) {
      fs.rmSync(chatbotPath, { recursive: true, force: true });
    }

    // Remove model from ./model/{userId}/{chatbotId}
    const modelPath = path.resolve(__dirname, "../../../model");
    const chatbotModelPath = path.join(modelPath, userId.toString(), chatbot.id.toString());

    if (fs.existsSync(chatbotModelPath)) {
      fs.rmSync(chatbotModelPath, { recursive: true, force: true });
    }

    // Check if removal was successful
    successRemoveModelandStorage = !fs.existsSync(chatbotPath) && !fs.existsSync(chatbotModelPath);
  }

  if (!successRemoveModelandStorage) {
    return c.json(
      {
        message: "Failed to remove model and storage",
      },
      HttpStatusCodes.INTERNAL_SERVER_ERROR,
    );
  }

  await db.delete(chatbots).where(eq(chatbots.id, id));

  return c.json(
    { message: "Chatbot deleted successfully" },
    HttpStatusCodes.OK,
  );
};
