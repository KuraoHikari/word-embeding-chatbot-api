import { and, count, eq } from "drizzle-orm";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";

import type { AppRouteHandler } from "@/lib/types";

import db from "@/db";
import { loadPDFIntoPinecone } from "@/db/pinecone";
import { chatbots, contacts, conversations, defaultSystemPrompt, messages } from "@/db/schema";
import env from "@/env";
import { ZOD_ERROR_CODES, ZOD_ERROR_MESSAGES } from "@/lib/constants";
import { TrainingError } from "@/lib/error";
import { sendTrainingRequestWithoutRetry } from "@/lib/send-training-request-without-retry";

import type { CreateChatbot, DeleteChatbot, GetChatbot, ListChatbot, PatchChatbot } from "./chatbots.routes";

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

export const getOne: AppRouteHandler<GetChatbot> = async (c) => {
  const userId = c.get("userId");

  if (!userId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }

  const { id } = c.req.valid("param");

  // Get chatbot details
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
      { message: HttpStatusPhrases.NOT_FOUND },
      HttpStatusCodes.NOT_FOUND,
    );
  }

  // Use Promise.all to fetch all counts in parallel
  const [conversationsResult, messagesResult, aiResponsesResult] = await Promise.all([
    // Count total conversations for this chatbot
    db
      .select({ count: count() })
      .from(conversations)
      .where(eq(conversations.chatbotId, id)),

    // Count total messages for this chatbot (through conversations)
    db
      .select({ count: count() })
      .from(messages)
      .innerJoin(conversations, eq(messages.conversationId, conversations.id))
      .where(eq(conversations.chatbotId, id)),

    // Count total AI responses (messages where isBot = true)
    db
      .select({ count: count() })
      .from(messages)
      .innerJoin(conversations, eq(messages.conversationId, conversations.id))
      .where(and(eq(conversations.chatbotId, id), eq(messages.isBot, true)),
      ),
  ]);

  return c.json(
    {
      ...chatbot,
      totalConversations: conversationsResult[0]?.count ?? 0,
      totalMessages: messagesResult[0]?.count ?? 0,
      totalAiResponses: aiResponsesResult[0]?.count ?? 0,
    },
    HttpStatusCodes.OK,
  );
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

  // Insert chatbot data into database
  // create test contact and conversation

  const [newChatbot] = await db.insert(chatbots).values({
    title: chatbotData.title,
    description: chatbotData.description,
    isPublic: chatbotData.isPublic,
    welcomeMessage: chatbotData.welcomeMessage,
    suggestionMessage: chatbotData.suggestionMessage,
    systemPrompt: chatbotData.systemPrompt ? chatbotData.systemPrompt : defaultSystemPrompt,
    aiModel: chatbotData.aiModel,
    isProposedModel: chatbotData.isProposedModel,
    embeddingModel: chatbotData.embeddingModel,
    temperature: Math.round((chatbotData.temperature ?? 0.3) * 100), // store as integer (e.g. 0.3 -> 30)
    maxTokens: chatbotData.maxTokens ?? 500,
    pdfTitle: pdf.name,
    pdfLink: pdf.name,
    userId: Number(userId),
  }).returning({
    id: chatbots.id,
    title: chatbots.title,
    aiModel: chatbots.aiModel,
  });

  const testContact = await db.insert(contacts).values({
    userId: Number(userId),
    name: "Test User",
    email: "test@example.com",
  }).returning({
    id: contacts.id,
  });

  await db.insert(conversations).values({
    userId: Number(userId),
    chatbotId: newChatbot.id, // temporary, will update later
    contactId: testContact[0].id,
  });

  try {
    // Simpan data chatbot ke database

    // Proses PDF ke Pinecone
    const pdfBuffer = await pdf.arrayBuffer();
    const pdfBlob = new Blob([pdfBuffer], { type: pdf.type });

    if (chatbotData.embeddingModel === "pinecone") {
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
      form.append("modelType", chatbotData.embeddingModel);
      form.append("pdfTitle", pdf.name);

      // Kirim request ke Python server
      // Kirim request training dengan retry
      let traningType: "baseline-model" | "proposed-model" = "baseline-model";

      if (chatbotData.isProposedModel) {
        traningType = "proposed-model";
      }
      else {
        traningType = "baseline-model";
      }

      const trainingResponse = await sendTrainingRequestWithoutRetry(
        form,
        env.API_PASSWORD,
        traningType,
      );

      c.var.logger.info("Training request sent successfully", trainingResponse);

      c.var.logger.debug("Training response status:", trainingResponse);

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
  const formData = c.req.valid("form");
  const { pdf, ...updates } = formData;

  // Check if chatbot exists and belongs to user
  const existingChatbot = await db.query.chatbots.findFirst({
    where(fields, operators) {
      return operators.and(
        operators.eq(fields.id, id),
        operators.eq(fields.userId, userId),
      );
    },
  });

  if (!existingChatbot) {
    return c.json(
      { message: HttpStatusPhrases.NOT_FOUND },
      HttpStatusCodes.NOT_FOUND,
    );
  }

  // Remove undefined/empty values from updates
  const cleanUpdates: Record<string, any> = {};
  for (const [key, value] of Object.entries(updates)) {
    if (value !== undefined && value !== "") {
      cleanUpdates[key] = value;
    }
  }

  // Convert temperature back to integer if provided
  if (cleanUpdates.temperature !== undefined) {
    cleanUpdates.temperature = Math.round(cleanUpdates.temperature * 100);
  }

  // Check if there are any updates
  if (Object.keys(cleanUpdates).length === 0 && !pdf) {
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

  try {
    // If PDF is provided, we need to retrain the model
    if (pdf && pdf instanceof File) {
      // Validate PDF size
      if (pdf.size > 10 * 1024 * 1024) {
        return c.json(
          { message: "PDF file size exceeds 10MB" },
          HttpStatusCodes.BAD_REQUEST,
        );
      }

      // Remove old model and storage directories (only for non-pinecone models)
      if (existingChatbot.embeddingModel !== "pinecone") {
        const storagePath = path.resolve(__dirname, "../../../storage");
        const chatbotPath = path.join(storagePath, userId.toString(), existingChatbot.id.toString());

        if (fs.existsSync(chatbotPath)) {
          fs.rmSync(chatbotPath, { recursive: true, force: true });
        }

        const modelPath = path.resolve(__dirname, "../../../model");
        const chatbotModelPath = path.join(modelPath, userId.toString(), existingChatbot.id.toString());

        if (fs.existsSync(chatbotModelPath)) {
          fs.rmSync(chatbotModelPath, { recursive: true, force: true });
        }
      }

      // Add PDF metadata to updates
      cleanUpdates.pdfTitle = pdf.name;
      cleanUpdates.pdfLink = pdf.name;

      // Process PDF for training
      const pdfBuffer = await pdf.arrayBuffer();
      const pdfBlob = new Blob([pdfBuffer], { type: pdf.type });

      // Determine embedding model (use existing if not updated)
      const embeddingModel = cleanUpdates.embeddingModel || existingChatbot.embeddingModel;

      if (embeddingModel === "pinecone") {
        // Upload to Pinecone
        await loadPDFIntoPinecone({
          pdfBlob,
          namespace: existingChatbot.id.toString(),
        });
      }
      else {
        const form = new FormData();
        form.append("pdf", pdfBlob, "document.pdf");
        form.append("userId", String(userId));
        form.append("chatbotId", String(existingChatbot.id));
        form.append("modelType", embeddingModel);
        form.append("pdfTitle", pdf.name);

        // Determine training type (use existing if not updated)
        const isProposedModel = cleanUpdates.isProposedModel !== undefined
          ? cleanUpdates.isProposedModel
          : existingChatbot.isProposedModel;

        const trainingType: "baseline-model" | "proposed-model" = isProposedModel
          ? "proposed-model"
          : "baseline-model";

        const trainingResponse = await sendTrainingRequestWithoutRetry(
          form,
          env.API_PASSWORD,
          trainingType,
        );

        c.var.logger.info("Training request sent successfully for update");

        if (!trainingResponse.ok) {
          const errorBody = await trainingResponse.json();
          throw new TrainingError("Training failed during update", errorBody);
        }
      }
    }

    // Update chatbot in database
    const [updatedChatbot] = await db.update(chatbots)
      .set(cleanUpdates)
      .where(eq(chatbots.id, id))
      .returning();

    if (!updatedChatbot) {
      return c.json(
        { message: HttpStatusPhrases.INTERNAL_SERVER_ERROR },
        HttpStatusCodes.INTERNAL_SERVER_ERROR,
      );
    }

    return c.json(
      { message: "Chatbot updated successfully" },
      HttpStatusCodes.OK,
    );
  }
  catch (error) {
    console.error("Error updating chatbot:", error);
    return c.json(
      {
        message: HttpStatusPhrases.INTERNAL_SERVER_ERROR,
        details: error instanceof Error ? error.message : "Unknown error",
      },
      HttpStatusCodes.INTERNAL_SERVER_ERROR,
    );
  }
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
  if (chatbot?.embeddingModel === "pinecone") {
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
