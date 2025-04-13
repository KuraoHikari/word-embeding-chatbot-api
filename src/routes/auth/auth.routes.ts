import { createRoute, z } from "@hono/zod-openapi";
import * as HttpStatusCodes from "stoker/http-status-codes";
import { jsonContent, jsonContentRequired } from "stoker/openapi/helpers";
import { createErrorSchema } from "stoker/openapi/schemas";

import { loginUserSchema, registerUserSchema } from "@/db/schema";
import { conflictSchema, createdSchema, tooManyRequestsSchema, unauthorizedSchema } from "@/lib/constants";

const tags = ["Auth"];

const path = "auth";

export const register = createRoute({
  path: `/${path}/register`,
  method: "post",
  tags,
  request: {
    body: jsonContentRequired(registerUserSchema, "The user to register"),
  },
  responses: {
    [HttpStatusCodes.CREATED]: jsonContent(
      createdSchema,
      "The registration response",
    ),
    [HttpStatusCodes.UNPROCESSABLE_ENTITY]: jsonContent(
      createErrorSchema(registerUserSchema),
      "The validation error(s)",
    ),
    [HttpStatusCodes.CONFLICT]: jsonContent(
      conflictSchema,
      "The user already exists",
    ),
  },
});

export const login = createRoute({
  path: `/${path}/login`,
  method: "post",
  tags,
  request: {
    body: jsonContentRequired(loginUserSchema, "The user to login"),
  },
  responses: {
    [HttpStatusCodes.OK]: jsonContent(
      z.object({
        access_token: z.string(),
      }),
      "The login response",
    ),
    [HttpStatusCodes.UNPROCESSABLE_ENTITY]: jsonContent(
      createErrorSchema(registerUserSchema),
      "The validation error(s)",
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      unauthorizedSchema,
      "The invalid credentials",
    ),
  },
});

export const getUser = createRoute({
  path: `/${path}/me`,
  method: "get",
  tags,
  request: {
    headers: z.object({
      authorization: z.string().optional(),
    }),
  },
  responses: {
    [HttpStatusCodes.OK]: jsonContent(
      z.object({
        id: z.number(),
        email: z.string(),
        name: z.string(),
      }),
      "The user",
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      unauthorizedSchema,
      "The invalid credentials",
    ),
    [HttpStatusCodes.TOO_MANY_REQUESTS]: jsonContent(
      tooManyRequestsSchema,
      "The rate limit errors",
    ),
  },
});

export type LoginRoute = typeof login;
export type RegisterRoute = typeof register;
export type GetUserRoute = typeof getUser;
