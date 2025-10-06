import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";

import type { AppRouteHandler, QueryBaselineModelResponse, QueryProposedModelResponse } from "@/lib/types";

import db from "@/db";
import { messages } from "@/db/schema";
import env from "@/env";
import { sendQueryRequestWithoutRetry } from "@/lib/send-query-request-without-retry";

import type { CreateRoute } from "./messages.routes";

export const create: AppRouteHandler<CreateRoute> = async (c) => {
  const contactId = c.get("contactId");
  if (!contactId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }
  const message = c.req.valid("json");

  // find chatbot by id
  const chatbot = await db.query.chatbots.findFirst({
    where(fields, operators) {
      return operators.eq(fields.id, message.chatbotId);
    },
  });
  if (!chatbot) {
    return c.json(
      { message: HttpStatusPhrases.NOT_FOUND },
      HttpStatusCodes.NOT_FOUND,
    );
  }

  const newMessage = await db.insert(messages).values({
    userId: chatbot.userId,
    conversationId: message.conversationId,
    isBot: false,
    text: message.text,
  }).returning();

  if (!newMessage) {
    return c.json(
      { message: HttpStatusPhrases.INTERNAL_SERVER_ERROR },
      HttpStatusCodes.INTERNAL_SERVER_ERROR,
    );
  }
  // Example usage in a handler
  const response = await sendQueryRequestWithoutRetry(
    {
      query: message.text,
      userId: String(chatbot.userId),
      chatbotId: String(chatbot.id),
      modelType: (chatbot.embeddingModel === "fasttext" || chatbot.embeddingModel === "word2vec")
        ? chatbot.embeddingModel
        : "fasttext",
      promptTemplate: chatbot.systemPrompt || "",
      pdfTitle: chatbot.pdfTitle,
      topK: 3,
      useGPT: true,
      gptModel: chatbot.aiModel || "gpt-3.5-turbo",
      similarityThreshold: 0.4,
      includeRAGAS: true,
      maxToken: chatbot.maxTokens || 500,
      temperature: chatbot.temperature ? chatbot.temperature / 100 : 0.3,
      isProposedModel: chatbot.isProposedModel || false,
    },
    env.API_PASSWORD,
  );
  console.log("🚀 ~ create ~ response:", response);

  if (!response.ok) {
    return c.json(
      { message: HttpStatusPhrases.INTERNAL_SERVER_ERROR },
      HttpStatusCodes.INTERNAL_SERVER_ERROR,
    );
  }

  // if isProposedModel use QueryProposedModelResponse for result type
  let results: QueryProposedModelResponse | QueryBaselineModelResponse;

  if (chatbot.isProposedModel) {
    results = await response.json() as QueryProposedModelResponse;
  }
  else {
    results = await response.json() as QueryBaselineModelResponse;
  }

  if (!results) {
    return c.json(
      { message: HttpStatusPhrases.INTERNAL_SERVER_ERROR },
      HttpStatusCodes.INTERNAL_SERVER_ERROR,
    );
  }

  const botAnswer = await db.insert(messages).values({
    userId: chatbot.userId,
    conversationId: message.conversationId,
    isBot: true,
    text: results.gpt_generation?.answer || "Maaf, saya tidak dapat menjawab pertanyaan Anda saat ini.",
  }).returning();

  if (!botAnswer) {
    return c.json(
      { message: HttpStatusPhrases.INTERNAL_SERVER_ERROR },
      HttpStatusCodes.INTERNAL_SERVER_ERROR,
    );
  }

  return c.json({ ...botAnswer[0], results }, HttpStatusCodes.CREATED);
};
