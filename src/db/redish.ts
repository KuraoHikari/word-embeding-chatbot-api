// src/lib/redis-store.ts
import type { ClientRateLimitInfo, Store } from "hono-rate-limiter";

import { Redis } from "@upstash/redis/cloudflare";

import env from "@/env";

let redisInstance: Redis | null = null;
let redisStoreInstance: UpstashRedisStore | null = null;

export class UpstashRedisStore implements Store {
  constructor(private readonly redis: Redis) {}

  async get(key: string): Promise<ClientRateLimitInfo | undefined> {
    const result = await this.redis.get<number>(key);
    if (result === null)
      return undefined;

    return {
      totalHits: result,
      resetTime: new Date(Date.now() + 60 * 60 * 1000),
    };
  }

  async set(key: string, value: ClientRateLimitInfo, ttlMs: number): Promise<void> {
    await this.redis.setex(key, Math.ceil(ttlMs / 1000), value.totalHits);
  }

  async increment(key: string): Promise<ClientRateLimitInfo> {
    const totalHits = await this.redis.incr(key);
    return {
      totalHits,
      resetTime: new Date(Date.now() + 60 * 60 * 1000),
    };
  }

  async decrement(key: string): Promise<void> {
    await this.redis.decr(key);
  }

  async resetKey(key: string): Promise<void> {
    await this.redis.del(key);
  }
}

export function getRedisInstance(): Redis {
  if (!redisInstance) {
    redisInstance = new Redis({
      url: env.UPSTASH_REDIS_REST_URL,
      token: env.UPSTASH_REDIS_REST_TOKEN,
    });
  }
  return redisInstance;
}

export function getRedisStore(): UpstashRedisStore {
  if (!redisStoreInstance) {
    redisStoreInstance = new UpstashRedisStore(getRedisInstance());
  }
  return redisStoreInstance;
}
