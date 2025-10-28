// src/middleware/authMiddleware.ts
import type { MiddlewareHandler } from "hono";

import { verify } from "hono/jwt";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";

import env from "@/env";

export const authMiddleware: MiddlewareHandler = async (c, next) => {
  const authHeader = c.req.header("Authorization");
  console.log("Auth Middleware Invoked");

  // // console the body
  // console.log(c.req.valid("form"));

  // const formData = c.req.valid("form");
  // const { pdf, ...chatbotData } = formData;

  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
  }

  const token = authHeader.slice(7);

  try {
    const payload = await verify(token, env.ACCESS_TOKEN_SECRET);
    c.set("userId", payload.sub); // Simpan user ID di context
    await next(); // Lanjut ke handler berikutnya
  }
  catch {
    return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
  }
};

export const authMiddlewarePublicContact: MiddlewareHandler = async (c, next) => {
  const authHeader = c.req.header("Authorization");

  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
  }

  const token = authHeader.slice(7);

  try {
    const payload = await verify(token, env.ACCESS_TOKEN_SECRET_PUBLIC);

    c.set("contactId", payload.sub); // Simpan user ID di context
    await next(); // Lanjut ke handler berikutnya
  }
  catch {
    return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
  }
};
