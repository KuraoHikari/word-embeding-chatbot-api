import { createRoute, z } from "@hono/zod-openapi";
import * as HttpStatusCodes from "stoker/http-status-codes";
import { jsonContent, jsonContentRequired } from "stoker/openapi/helpers";
import { createErrorSchema } from "stoker/openapi/schemas";

import { loginUserSchema, registerUserSchema } from "@/db/schema";
import { conflictSchema, createdSchema, tooManyRequestsSchema, unauthorizedSchema } from "@/lib/constants";

const tags = ["Auth"];

const path = "auth";

const trendPointSchema = z.object({
  period: z.string(),
  count: z.number().int().nonnegative(),
});

const trendSeriesSchema = z.object({
  daily: z.array(trendPointSchema),
  weekly: z.array(trendPointSchema),
  monthly: z.array(trendPointSchema),
});

const dashboardDetailSchema = z.object({
  summary: z.object({
    totalChatbots: z.number().int().nonnegative(),
    totalConversations: z.number().int().nonnegative(),
    totalMessages: z.number().int().nonnegative(),
    totalContacts: z.number().int().nonnegative(),
    totalAiResponses: z.number().int().nonnegative(),
    avgMessagesPerConversation: z.number().nonnegative(),
  }),
  trends: z.object({
    incomingMessages: trendSeriesSchema,
    newConversations: trendSeriesSchema,
    newContacts: trendSeriesSchema,
    autoReplyRatio: z.number().min(0).max(1),
  }),
  performance: z.object({
    topChatbotsByConversations: z.array(
      z.object({
        chatbotId: z.number().int().nonnegative(),
        title: z.string(),
        conversations: z.number().int().nonnegative(),
      }),
    ),
    topChatbotsByMessages: z.array(
      z.object({
        chatbotId: z.number().int().nonnegative(),
        title: z.string(),
        messages: z.number().int().nonnegative(),
      }),
    ),
    topChatbotsByAvgUserMessageLength: z.array(
      z.object({
        chatbotId: z.number().int().nonnegative(),
        title: z.string(),
        avgLength: z.number().nonnegative(),
      }),
    ),
  }),
});

export const register = createRoute({
  path: `/${path}/register`,
  method: "post",
  tags,
  request: {
    body: jsonContentRequired(registerUserSchema, "The user to register"),
  },
  responses: {
    [HttpStatusCodes.CREATED]: jsonContent(createdSchema, "The registration response"),
    [HttpStatusCodes.UNPROCESSABLE_ENTITY]: jsonContent(createErrorSchema(registerUserSchema), "The validation error(s)"),
    [HttpStatusCodes.CONFLICT]: jsonContent(conflictSchema, "The user already exists"),
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
        user: z.object({
          id: z.number(),
          email: z.string(),
          name: z.string(),
        }),
        access_token: z.string(),
      }),
      "The login response",
    ),
    [HttpStatusCodes.UNPROCESSABLE_ENTITY]: jsonContent(createErrorSchema(registerUserSchema), "The validation error(s)"),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(unauthorizedSchema, "The invalid credentials"),
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
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(unauthorizedSchema, "The invalid credentials"),
    [HttpStatusCodes.TOO_MANY_REQUESTS]: jsonContent(tooManyRequestsSchema, "The rate limit errors"),
  },
});

export const getDetailDashboard = createRoute({
  path: `/${path}/me/dashboard`,
  method: "get",
  tags,
  request: {
    headers: z.object({
      authorization: z.string().optional(),
    }),
  },
  responses: {
    [HttpStatusCodes.OK]: jsonContent(dashboardDetailSchema, "The dashboard details"),
   [HttpStatusCodes.UNAUTHORIZED]: jsonContent(unauthorizedSchema, "The invalid credentials"),
     [HttpStatusCodes.TOO_MANY_REQUESTS]: jsonContent(tooManyRequestsSchema, "The rate limit errors"),
 }  ,
  });
  
 export type LoginRoute = typeof login;
export type RegisterRoute = typeof register;
export type GetUserRoute = typeof getUser;
export type GetDetailDashboardRoute = typeof getDetailDashboard;
