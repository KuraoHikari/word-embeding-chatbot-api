import { rateLimiter } from "hono-rate-limiter";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";

import type { AppBindings } from "@/lib/types";

import { getRedisStore } from "@/db/redish";
import env from "@/env";

export function createLimiter({
  limit,
  windowMinutes,
  key = "",
  isPublicContact = false,
}: {
  limit: number;

  windowMinutes: number;
  key?: string;
  isPublicContact?: boolean;
}) {
  const isDevelopment = env.NODE_ENV === "development";

  return rateLimiter<AppBindings>({
    windowMs: windowMinutes * 60 * 1000,
    limit,
    store: isDevelopment ? undefined : getRedisStore(), // Use the singleton store instance
    standardHeaders: "draft-7",
    keyGenerator: (c) => {
      const ip = c.req.header("x-forwarded-for") || c.req.header("cf-connecting-ip") || "anonymous";
      const id = isPublicContact ? c.get("contactId") : c.get("userId") || "guest";
      const pathKey = key || c.req.path;
      return `${id}_${ip}_${pathKey}`;
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

export const chatbotsLimiter = createLimiter({
  limit: 5,
  windowMinutes: 1,
});
