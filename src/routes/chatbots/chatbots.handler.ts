import { eq } from "drizzle-orm";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";
import { v4 as uuidv4 } from "uuid";

import type { AppRouteHandler } from "@/lib/types";

import db from "@/db";
import { loadPDFIntoPinecone } from "@/db/pinecone";
import { chatbots } from "@/db/schema";
import env from "@/env";
import { TrainingError } from "@/lib/error";
import { sendTrainingRequestWithRetry } from "@/lib/send-training-request-with-retry";
import { sendTrainingRequestWithoutRetry } from "@/lib/send-training-request-without-retry";

import type { CreateChatbot, ListChatbot } from "./chatbots.routes";

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
