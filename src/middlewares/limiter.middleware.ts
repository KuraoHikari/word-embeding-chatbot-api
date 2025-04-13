import { rateLimiter } from "hono-rate-limiter";

import type { AppBindings } from "@/lib/types";

import { getRedisStore } from "@/db/redish";

import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";


export function createLimiter({
  limit,
  windowMinutes,
  key = "",
}: {
  limit: number;

  windowMinutes: number;
  key?: string;
}) {
  return rateLimiter<AppBindings>({
    windowMs: windowMinutes * 60 * 1000,
    limit,
    store: getRedisStore(), // Use the singleton store instance
    standardHeaders: "draft-7",
    keyGenerator: (c) => {
      const ip = c.req.header("x-forwarded-for") || c.req.header("cf-connecting-ip") || "anonymous";
      const userId = c.get("userId") || "guest";
      const pathKey = key || c.req.path;
      return `${userId}_${ip}_${pathKey}`;
    },
    handler: (c) => {
      return c.json(
        {
          message: HttpStatusPhrases.TOO_MANY_REQUESTS,
        },
        HttpStatusCodes.TOO_MANY_REQUESTS,
      );
    },

    skip: (_) => {
      // Skip rate limiting for certain conditions (optional)
      return false;
    },
  });
}

// Pre-configured limiters
export const authLimiter = createLimiter({
  limit: 10,
  windowMinutes: 15,
  key: "auth",
});

export const apiLimiter = createLimiter({
  limit: 100,
  windowMinutes: 60,
});

export const strictLimiter = createLimiter({
  limit: 5,
  windowMinutes: 1,
});
