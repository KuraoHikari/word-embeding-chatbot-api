import { eq } from "drizzle-orm";
import { OpenAI } from "openai";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";

import type { AppRouteHandler } from "@/lib/types";

import db from "@/db";
import { messages } from "@/db/schema";
import env from "@/env";
import { sendQueryRequestWithoutRetry } from "@/lib/send-query-request-without-retry";

import type { CreateRoute } from "./messages.routes";

const openai = new OpenAI({
  apiKey: env.OPEN_AI_API_KEY, // This is also the default, can be omitted
});

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
  // const newMessage = await db.insert(messages).values({
  //     userId: chatbot.userId,
  //     conversationId: message.conversationId,
  //     isBot: false,
  //     text: message.text,
  // }).returning();

  // if (!newMessage) {
  //     return c.json(
  //         { message: HttpStatusPhrases.INTERNAL_SERVER_ERROR },
  //         HttpStatusCodes.INTERNAL_SERVER_ERROR,
  //     );
  // }
  // Example usage in a handler
  const response = await sendQueryRequestWithoutRetry(
    {
      query: message.text,
      userId: String(chatbot.userId),
      chatbotId: String(chatbot.id),
      modelType: chatbot.embedingModel,
      pdfTitle: chatbot.pdfTitle,
      topK: 2,
      similarityThreshold: 0.8,
    },
    env.API_PASSWORD,
  );
  console.log("🚀 ~ constcreate:AppRouteHandler<CreateRoute>= ~ response:", response);

  const results = await response.json();

  if (!results) {
    return c.json(
      { message: HttpStatusPhrases.INTERNAL_SERVER_ERROR },
      HttpStatusCodes.INTERNAL_SERVER_ERROR,
    );
  }

  const prompt = "AI assistant is a professional and polite customer service. \nThe traits of AI include expert knowledge, helpfulness, cleverness, and articulateness. \nAI assistant provides clear, concise, and friendly responses without repeating unnecessary information or phrases such as \"Berdasarkan informasi yang diberikan sebelumnya.\", \"dalam konteks yang diberikan.\", \"dalam konteks yang tersedia.\". \nAI is a well-behaved and well-mannered individual. \nAI is always friendly, kind, and inspiring, and he is eager to provide vivid and thoughtful responses to the user. \nAI has the sum of all knowledge in their brain, and is able to accurately answer nearly any question about any topic in conversation. \nAI assistant make answer using Indonesian Language. \nAI assistant avoids sounding repetitive and ensures responses sound natural and tailored to each question. \nIf the context does not provide the answer to question, the AI assistant will say, \"Mohon Maaf, tapi saya tidak dapat menjawab pertanyaan tersebut saat ini.\". \nSTART CONTEXT BLOCK \n{{context}} \nEND OF CONTEXT BLOCK \nAI assistant will take into account any CONTEXT BLOCK that is provided in a conversation. \nIf the context does not provide the answer to question, the AI assistant will say, \"Mohon Maaf, tapi saya tidak dapat menjawab pertanyaan tersebut saat ini.\". \nAI assistant will not apologize for previous responses, but instead will indicated new information was gained. \nAI assistant will not invent anything that is not drawn directly from the context.";

  const promtFromDb = {
    role: "system" as const,
    content: prompt.replace("{{context}}", results.results),
  };

  const askAi = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [
      promtFromDb,
      {
        role: "user" as const,
        content: message.text,
      },
    ],
  });
  return c.json({
    ...askAi.choices[0].message,
    results,

    promtFromDb,
  });
};
