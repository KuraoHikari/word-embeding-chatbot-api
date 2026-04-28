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

  const userId = typeof payload.sub === "string" ? Number(payload.sub) : payload.sub;
  if (typeof userId !== "number" || !Number.isFinite(userId)) {
   return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
  }

  c.set("userId", userId); // Simpan user ID di context
  await next(); // Lanjut ke handler berikutnya
 } catch {
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

  const contactId = typeof payload.sub === "string" ? Number(payload.sub) : payload.sub;
  if (typeof contactId !== "number" || !Number.isFinite(contactId)) {
   return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
  }

  c.set("contactId", contactId); // Simpan user ID di context
  await next(); // Lanjut ke handler berikutnya
 } catch {
  return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
 }
};
