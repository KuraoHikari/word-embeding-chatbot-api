import { createClient } from "@libsql/client";
import env from "../src/env";

async function main() {
 const client = createClient({
  url: env.DATABASE_URL,
  authToken: env.DATABASE_AUTH_TOKEN,
 });

 // 1. Add sender_role column
 try {
  await client.execute(`ALTER TABLE messages ADD COLUMN sender_role TEXT NOT NULL DEFAULT 'contact'`);
  console.log("✅ Column sender_role added");
 } catch (e: any) {
  if (e.message?.includes("duplicate column")) {
   console.log("ℹ️  Column sender_role already exists");
  } else {
   throw e;
  }
 }

 // 2. Backfill existing bot messages
 const updated = await client.execute(`UPDATE messages SET sender_role = 'bot' WHERE is_bot = 1`);
 console.log(`✅ Updated ${updated.rowsAffected} bot messages to senderRole='bot'`);

 // 3. Mark migration as applied in drizzle tracker
 try {
  await client.execute({
   sql: `INSERT INTO __drizzle_migrations (hash, created_at) VALUES (?, ?)`,
   args: ["0002_yellow_king_cobra", Date.now()],
  });
  console.log("✅ Migration 0002 tracked in __drizzle_migrations");
 } catch (e: any) {
  console.log("ℹ️  Migration tracking:", e.message);
 }

 console.log("Done!");
}

main().catch(console.error);
