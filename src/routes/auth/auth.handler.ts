import { verify } from "hono/jwt";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";

import type { AppRouteHandler } from "@/lib/types";

import db from "@/db";
import { users } from "@/db/schema";
import env from "@/env";
import { hash, verify as verifyHash } from "@/lib/hashing";
import { generateAccessToken } from "@/lib/token";

import type { GetUserRoute, LoginRoute, RegisterRoute } from "./auth.routes";

export const login: AppRouteHandler<LoginRoute> = async (c) => {
  const { email, password } = c.req.valid("json");

  // Check if user exists
  const user = await db.query.users.findFirst({
    where(fields, operators) {
      return operators.eq(fields.email, email);
    },
  });

  if (!user) {
    return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
  }

  // Check if password is correct
  const isPasswordCorrect = await verifyHash(
    user.password,
    password,
  );

  if (!isPasswordCorrect) {
    return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
  }

  // Generate access token
  const token = await generateAccessToken(user.id);

  return c.json({ access_token: token }, HttpStatusCodes.OK);
};

export const register: AppRouteHandler<RegisterRoute> = async (c) => {
  const { email, password, name } = c.req.valid("json");

  // Check if user already exists
  const existingUser = await db.query.users.findFirst({
    where(fields, operators) {
      return operators.eq(fields.email, email);
    },
  });

  if (existingUser) {
    return c.json({ message: HttpStatusPhrases.CONFLICT }, HttpStatusCodes.CONFLICT);
  }

  // hash password
  const hashedPassword = await hash(password);

  // Create user
  await db.insert(users).values({ email, password: hashedPassword, name }).returning();

  return c.json({
    message: HttpStatusPhrases.CREATED,
  }, HttpStatusCodes.CREATED);
};

export const getUser: AppRouteHandler<GetUserRoute> = async (c) => {
  const userId = c.get("userId");

  if (!userId) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }

  const user = await db.query.users.findFirst({
    where(fields, operators) {
      return operators.eq(fields.id, userId);
    },
    columns: {
      id: true,
      email: true,
      name: true,
    },
  });

  if (!user) {
    return c.json(
      { message: HttpStatusPhrases.UNAUTHORIZED },
      HttpStatusCodes.UNAUTHORIZED,
    );
  }

  return c.json(user, HttpStatusCodes.OK);
};
