import { getRedisInstance } from "@/db/redish";

const ONLINE_KEY_PREFIX = "online:";
const ONLINE_TTL_SECONDS = 300; // 5 minutes heartbeat window

/**
 * Mark a user as online in Redis.
 */
export async function setOnline(userId: number): Promise<void> {
  const redis = getRedisInstance();
  await redis.set(`${ONLINE_KEY_PREFIX}${userId}`, "1", { ex: ONLINE_TTL_SECONDS });
}

/**
 * Remove a user's online status from Redis.
 */
export async function setOffline(userId: number): Promise<void> {
  const redis = getRedisInstance();
  await redis.del(`${ONLINE_KEY_PREFIX}${userId}`);
}

/**
 * Check whether a user is currently online.
 */
export async function isOnline(userId: number): Promise<boolean> {
  const redis = getRedisInstance();
  const result = await redis.get(`${ONLINE_KEY_PREFIX}${userId}`);
  return result !== null;
}
