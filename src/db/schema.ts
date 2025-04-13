import type { SQL } from "drizzle-orm";
import type { AnySQLiteColumn } from "drizzle-orm/sqlite-core";

import { sql } from "drizzle-orm";
import { integer, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";

const id = {
  id: integer("id", { mode: "number" })
    .primaryKey({ autoIncrement: true }),
};

const timestamps = {
  createdAt: integer("created_at", { mode: "timestamp" })
    .$defaultFn(() => new Date()),
  updatedAt: integer("updated_at", { mode: "timestamp" })
    .$defaultFn(() => new Date())
    .$onUpdate(() => new Date()),
};

export const tasks = sqliteTable("tasks", {
  ...id,
  name: text("name")
    .notNull(),
  done: integer("done", { mode: "boolean" })
    .notNull()
    .default(false),
  ...timestamps,
});

// custom lower function
export function lower(email: AnySQLiteColumn): SQL {
  return sql`lower(${email})`;
}

export const users = sqliteTable("users", {
  ...id,
  name: text("name")
    .notNull(),
  password: text("password")
    .notNull(),
  email: text("email").notNull(),
  ...timestamps,
}, table => ({
  emailUniqueIndex: uniqueIndex("emailUniqueIndex").on(lower(table.email)),
}));

export const selectTasksSchema = createSelectSchema(tasks);

export const insertTasksSchema = createInsertSchema(
  tasks,
  {
    name: schema => schema.name.min(1).max(500),
  },
).required({
  done: true,
}).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const registerUserSchema = createInsertSchema(
  users,
  {
    name: schema => schema.name.min(1).max(500),
    email: schema => schema.email.email().min(1).max(500),
    password: schema => schema.password.min(8).max(500),
  },
).required({
  email: true,
  password: true,
  name: true,
}).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const loginUserSchema = registerUserSchema.omit({
  name: true,
});

export const patchTasksSchema = insertTasksSchema.partial();
