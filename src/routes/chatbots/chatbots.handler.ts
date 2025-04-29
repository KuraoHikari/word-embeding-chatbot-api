import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";

import type { AppRouteHandler } from "@/lib/types";

import db from "@/db";
import { chatbots } from "@/db/schema";

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
  console.log("🚀 ~ constcreate:AppRouteHandler<CreateChatbot>= ~ userId:", userId);

  if (!userId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }

  const formData = c.req.valid("form");
  const { pdf, ...chatbotData } = formData;
  console.log("🚀 ~ constcreate:AppRouteHandler<CreateChatbot>= ~ pdf:", pdf);

  c.var.logger.info("Form data received:", pdf);

  if (pdf) {
    const pdfBuffer = await pdf.arrayBuffer();
    console.log("🚀 ~ constcreate:AppRouteHandler<CreateChatbot>= ~ pdfBuffer:", pdfBuffer);

    const pdfBlob = new Blob([pdfBuffer], { type: "application/pdf" });

    console.log("🚀 ~ constcreate:AppRouteHandler<CreateChatbot>= ~ pdfBlob:", pdfBlob);

    // Lakukan sesuatu dengan file PDF, misalnya simpan ke storage
    // Simpan file ke storage
  }

  if (!chatbotData) {
    return c.json(
      { message: "Chatbot data is required" },
      HttpStatusCodes.BAD_REQUEST,
    );
  }
  // Simpan data ke database
  //   await db.insert(chatbots).values({
  //     ...chatbotData,
  //     userId,
  //   });

  // Parse the JSON data

  return c.json({
    message: HttpStatusPhrases.CREATED,
  }, HttpStatusCodes.CREATED);
};
