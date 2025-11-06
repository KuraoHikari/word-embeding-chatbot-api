import type { SQL } from "drizzle-orm";
import type { AnySQLiteColumn } from "drizzle-orm/sqlite-core";

import { z } from "@hono/zod-openapi";
import { relations, sql } from "drizzle-orm";
import { integer, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";

export const defaultSystemPrompt = `AI assistant is a professional and polite customer service work at PT. Omni Hottilier representative. \n
The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness. \n
AI assistant provides clear, concise, and friendly responses without repeating unnecessary information or phrases such as "Berdasarkan informasi yang diberikan sebelumnya.", "dalam konteks yang diberikan.", "dalam konteks yang tersedia.". \n
AI is a well-behaved and well-mannered individual. \n
AI is always friendly, kind, and inspiring, and he is eager to provide vivid and thoughtful responses to the user. \n
AI has the sum of all knowledge in their brain, and is able to accurately answer nearly any question about any topic in conversation. \n
AI assistant make answer using Indonesian Language. \n
AI assistant avoids sounding repetitive and ensures responses sound natural and tailored to each question. \n
If the context does not provide the answer to question, the AI assistant will say, "Mohon Maaf, tapi saya tidak dapat menjawab pertanyaan tersebut saat ini.". \n
AI assistant will take into account any CONTEXT BLOCK that is provided in a conversation. \n
AI assistant will not apologize for previous responses, but instead will indicated new information was gained. \n
AI assistant will not invent anything that is not drawn directly from the context.
`;

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
  name: text("name").notNull(),
  password: text("password").notNull(),
  email: text("email").notNull(),
  ...timestamps,
}, table => ({
  emailUniqueIndex: uniqueIndex("emailUniqueIndex").on(lower(table.email)),
}));

// Chatbots Table
export const chatbots = sqliteTable("chatbots", {
  ...id,
  title: text("title").notNull(),
  description: text("description"),
  isPublic: integer("is_public", { mode: "boolean" }).notNull().default(false),
  welcomeMessage: text("welcome_message").notNull(),
  suggestionMessage: text("suggestion_message").notNull(),
  systemPrompt: text("system_prompt").notNull().default(defaultSystemPrompt),
  aiModel: text("ai_model").notNull().default("gpt-3.5-turbo"),
  isProposedModel: integer("is_proposed_model", { mode: "boolean" }).notNull().default(true),
  embeddingModel: text("embedding_model").notNull().default("fastext"),
  temperature: integer("temperature").notNull().default(30),
  maxTokens: integer("max_tokens").notNull().default(500),
  pdfTitle: text("pdf_title").notNull(),
  pdfLink: text("pdf_link").notNull(),
  userId: integer("user_id").notNull().references(() => users.id),
  ...timestamps,
});

// Contacts Table
export const contacts = sqliteTable("contacts", {
  ...id,
  name: text("name").notNull(),
  email: text("email").notNull(),
  phone: text("phone"),
  userId: integer("user_id").notNull().references(() => users.id),
  ...timestamps,
});

// Conversations Table
export const conversations = sqliteTable("conversations", {
  ...id,
  userId: integer("user_id").notNull().references(() => users.id),
  chatbotId: integer("chatbot_id").notNull().references(() => chatbots.id),
  contactId: integer("contact_id").notNull().references(() => contacts.id),
  ...timestamps,
});

// Messages Table
export const messages = sqliteTable("messages", {
  ...id,
  text: text("text").notNull(),
  conversationId: integer("conversation_id").notNull().references(() => conversations.id),
  userId: integer("user_id").notNull().references(() => users.id),
  isBot: integer("is_bot", { mode: "boolean" }).notNull().default(false),
  ...timestamps,
});

// Model Responses Table - for storing both proposed and baseline model data
export const modelResponses = sqliteTable("model_responses", {
  ...id,
  messageId: integer("message_id")
    .notNull()
    .references(() => messages.id)
    .unique(), // Ensures a one-to-one relationship
  modelType: text("model_type", { enum: ["proposed", "baseline"] }).notNull(),

  // Common fields for both model types
  query: text("query").notNull(),
  processingTime: integer("processing_time").notNull(),
  results: text("results", { mode: "json" }).notNull(),
  metadata: text("metadata", { mode: "json" }).notNull(),

  // Fields specific to Proposed Model (nullable for baseline)
  complexityAnalysis: text("complexity_analysis", { mode: "json" }),
  searchPipeline: text("search_pipeline", { mode: "json" }),

  // Fields specific to Baseline Model (nullable for proposed)
  modelApproach: text("model_approach"),
  pipelineSteps: text("pipeline_steps", { mode: "json" }),

  // Optional fields that can exist in both models
  gptGeneration: text("gpt_generation", { mode: "json" }),
  ragasEvaluation: text("ragas_evaluation", { mode: "json" }),
  message: text("message"),

  userId: integer("user_id").notNull().references(() => users.id),
  chatbotId: integer("chatbot_id").notNull().references(() => chatbots.id),
  ...timestamps,
});

export const modelResponsesRelations = relations(modelResponses, ({ one }) => ({
  message: one(messages, {
    fields: [modelResponses.messageId],
    references: [messages.id],
  }),
  chatbot: one(chatbots, {
    fields: [modelResponses.chatbotId],
    references: [chatbots.id],
  }),
  user: one(users, {
    fields: [modelResponses.userId],
    references: [users.id],
  }),
}));

export const queryProposedModelResponses = sqliteTable("queryProposedModelResponses", {
  ...id,
  messageId: integer("message_id").notNull().references(() => messages.id),
  ...timestamps,
});

export const messagesRelations = relations(messages, ({ one }) => ({
  user: one(users, {
    fields: [messages.userId],
    references: [users.id],
  }),
  conversation: one(conversations, {
    fields: [messages.conversationId],
    references: [conversations.id],
  }),
  // New relation to model response
  modelResponse: one(modelResponses, {
    fields: [messages.id],
    references: [modelResponses.messageId],
  }),
}));

export const usersRelations = relations(users, ({ many }) => ({
  chatbots: many(chatbots),
  contacts: many(contacts),
  conversations: many(conversations),
  messages: many(messages),
  modelResponses: many(modelResponses),
}));

export const chatbotsRelations = relations(chatbots, ({ one, many }) => ({
  user: one(users, {
    fields: [chatbots.userId],
    references: [users.id],
  }),
  conversations: many(conversations),
  modelResponses: many(modelResponses),
}));

export const contactsRelations = relations(contacts, ({ one, many }) => ({
  user: one(users, {
    fields: [contacts.userId],
    references: [users.id],
  }),
  conversations: many(conversations),
}));

export const conversationsRelations = relations(conversations, ({ one, many }) => ({
  user: one(users, {
    fields: [conversations.userId],
    references: [users.id],
  }),
  chatbot: one(chatbots, {
    fields: [conversations.chatbotId],
    references: [chatbots.id],
  }),
  contact: one(contacts, {
    fields: [conversations.contactId],
    references: [contacts.id],
  }),
  messages: many(messages),
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

export const selectChatbotsSchema = createSelectSchema(chatbots);

export const detailChatbotSchema = selectChatbotsSchema.extend({
  totalMessages: z.number().min(0).optional().describe("Total messages sent in the chatbot"),
  totalConversations: z.number().min(0).optional().describe("Total conversations started with the chatbot"),
  totalAiResponses: z.number().min(0).optional().describe("Total AI responses generated by the chatbot"),
});

export const insertChatbotsSchema = createInsertSchema(
  chatbots,
  {
    title: schema => schema.title.min(1).max(100),
    description: schema => schema.description.min(1).max(500),
    isPublic: schema => schema.isPublic,
    welcomeMessage: schema => schema.welcomeMessage.min(1).max(1000),
    suggestionMessage: schema => schema.suggestionMessage.min(1).max(1000),
    systemPrompt: schema => schema.systemPrompt.min(1).max(2000),
    aiModel: schema => schema.aiModel.pipe(z.enum(["gpt-3.5-turbo", "gpt-4", "gpt-4-32k", "gpt-4o"])),
    isProposedModel: schema => schema.isProposedModel,
    embeddingModel: schema => schema.embeddingModel.pipe(z.enum(["word2vec", "fasttext", "pinecone"])),
    temperature: schema => schema.temperature.min(0).max(100),
    maxTokens: schema => schema.maxTokens.min(100).max(2000),
    pdfTitle: schema => schema.pdfTitle.min(1).max(500),
    pdfLink: schema => schema.pdfLink.min(1).max(500),
  },
).required({
  title: true,
  description: true,
  isPublic: true,
  welcomeMessage: true,
  suggestionMessage: true,
  aiModel: true,
  isProposedModel: true,
  embeddingModel: true,
  temperature: true,
  maxTokens: true,
  pdfTitle: true,
  pdfLink: true,
}).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
  userId: true,
});

export const patchChatbotsSchema = insertChatbotsSchema.omit({
  pdfTitle: true,
  pdfLink: true,
}).partial();

export const patchChatbotsFormSchema = patchChatbotsSchema.extend({
  pdf: z.instanceof(File).or(z.instanceof(Blob)).optional().describe("PDF file upload (optional - only if retraining)").openapi({ format: "binary" }),
  isPublic: z.preprocess(
    val => val === "true" ? true : val === "false" ? false : val === undefined ? undefined : val,
    z.boolean().optional(),
  ),
  isProposedModel: z.preprocess(
    val => val === "true" ? true : val === "false" ? false : val === undefined ? undefined : val,
    z.boolean().optional(),
  ),
  temperature: z.preprocess(
    val => typeof val === "string" && val !== "" ? Number.parseFloat(val) : val === "" ? undefined : val,
    z.number().min(0.0).max(1.0).optional(),
  ),
  maxTokens: z.preprocess(
    val => typeof val === "string" && val !== "" ? Number.parseInt(val, 10) : val === "" ? undefined : val,
    z.number().int().min(100).max(2000).optional(),
  ),
});

export const selectContactsSchema = createSelectSchema(contacts);

export const loginUserSchema = registerUserSchema.omit({
  name: true,
});

export const patchTasksSchema = insertTasksSchema.partial();

export const selectContactSchema = createSelectSchema(contacts);

export const createContactSchema = createInsertSchema(
  contacts,
  {
    name: schema => schema.name.min(1).max(500),
    email: schema => schema.email.email().min(1).max(500),
    phone: schema => schema.phone.min(1).max(500),
  },
).required({
  name: true,
  email: true,
}).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
  userId: true,
}).partial({
  phone: true,
});

export const patchContactSchema = createContactSchema.partial();
export const selectConversationsSchema = createSelectSchema(conversations);
export const insertConversationsSchema = createInsertSchema(
  conversations,
  {
    chatbotId: schema => schema.chatbotId.min(1).max(500),
  },
).required({
  chatbotId: true,
}).omit({
  id: true,
  contactId: true,
  userId: true,
  createdAt: true,
  updatedAt: true,
});

export const selectMessagesSchema = createSelectSchema(messages);
export const insertMessagesSchema = createInsertSchema(
  messages,
  {
    text: schema => schema.text.min(1).max(500),
    conversationId: schema => schema.conversationId.min(1).max(500),
  },
).required({
  text: true,
}).omit({
  id: true,
  userId: true,
  createdAt: true,
  updatedAt: true,
  isBot: true,
}).extend({
  // extend chatbotId required
  chatbotId: z.number().min(1).max(500),
});
