import { and, asc, count, desc, eq, isNotNull, sql } from "drizzle-orm";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";

import type { AppRouteHandler } from "@/lib/types";

import db from "@/db";
import { chatbots, contacts, conversations, messages, users } from "@/db/schema";
import { hash, verify as verifyHash } from "@/lib/hashing";
import { generateAccessToken } from "@/lib/token";

import type { GetDetailDashboardRoute, GetUserRoute, LoginRoute, RegisterRoute } from "./auth.routes";

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
  const isPasswordCorrect = await verifyHash(user.password, password);

  if (!isPasswordCorrect) {
    return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
  }

  // Generate access token
  const token = await generateAccessToken(user.id);

  return c.json(
    {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
      },
      access_token: token,
    },
    HttpStatusCodes.OK,
  );
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

  return c.json(
    {
      message: HttpStatusPhrases.CREATED,
    },
    HttpStatusCodes.CREATED,
  );
};

export const getUser: AppRouteHandler<GetUserRoute> = async (c) => {
  const userId = c.get("userId");

  if (!userId) {
    return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
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
    return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
  }

  return c.json(user, HttpStatusCodes.OK);
};

interface TrendPoint {
  period: string;
  count: number;
}
interface TrendSeries {
  daily: TrendPoint[];
  weekly: TrendPoint[];
  monthly: TrendPoint[];
}

function normalizeTrendRows(rows: Array<{ period: string | null; count: number }>): TrendPoint[] {
  return rows
    .filter(row => typeof row.period === "string" && row.period.length > 0)
    .map(row => ({
      period: row.period as string,
      count: Number(row.count ?? 0),
    }));
}

export const getDetailDashboard: AppRouteHandler<GetDetailDashboardRoute> = async (c) => {
  const userId = c.get("userId");

  if (!userId) {
    return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
  }

  const uid = Number(userId);
  if (!Number.isFinite(uid)) {
    return c.json({ message: HttpStatusPhrases.UNAUTHORIZED }, HttpStatusCodes.UNAUTHORIZED);
  }

  const [totalChatbotsResult, totalConversationsResult, totalMessagesResult, totalContactsResult, totalAiResponsesResult, autoReplyEnabledResult] = await Promise.all([
    db.select({ count: count() }).from(chatbots).where(eq(chatbots.userId, uid)),
    db.select({ count: count() }).from(conversations).where(eq(conversations.userId, uid)),
    db.select({ count: count() }).from(messages).where(eq(messages.userId, uid)),
    db.select({ count: count() }).from(contacts).where(eq(contacts.userId, uid)),
    db
      .select({ count: count() })
      .from(messages)
      .where(and(eq(messages.userId, uid), eq(messages.isBot, true))),
    db
      .select({ count: count() })
      .from(conversations)
      .where(and(eq(conversations.userId, uid), eq(conversations.autoReply, true))),
  ]);

  const totalChatbots = Number(totalChatbotsResult[0]?.count ?? 0);
  const totalConversations = Number(totalConversationsResult[0]?.count ?? 0);
  const totalMessages = Number(totalMessagesResult[0]?.count ?? 0);
  const totalContacts = Number(totalContactsResult[0]?.count ?? 0);
  const totalAiResponses = Number(totalAiResponsesResult[0]?.count ?? 0);
  const totalAutoReplyConversations = Number(autoReplyEnabledResult[0]?.count ?? 0);

  const avgMessagesPerConversation = totalConversations > 0 ? totalMessages / totalConversations : 0;

  const autoReplyRatio = totalConversations > 0 ? totalAutoReplyConversations / totalConversations : 0;

  // Ekspresi waktu
  const msgDayExpr = sql<string>`strftime('%Y-%m-%d', ${messages.createdAt}, 'unixepoch')`;
  const msgWeekExpr = sql<string>`strftime('%Y-%W', ${messages.createdAt}, 'unixepoch')`;
  const msgMonthExpr = sql<string>`strftime('%Y-%m', ${messages.createdAt}, 'unixepoch')`;

  const convDayExpr = sql<string>`strftime('%Y-%m-%d', ${conversations.createdAt}, 'unixepoch')`;
  const convWeekExpr = sql<string>`strftime('%Y-%W', ${conversations.createdAt}, 'unixepoch')`;
  const convMonthExpr = sql<string>`strftime('%Y-%m', ${conversations.createdAt}, 'unixepoch')`;

  const contactDayExpr = sql<string>`strftime('%Y-%m-%d', ${contacts.createdAt}, 'unixepoch')`;
  const contactWeekExpr = sql<string>`strftime('%Y-%W', ${contacts.createdAt}, 'unixepoch')`;
  const contactMonthExpr = sql<string>`strftime('%Y-%m', ${contacts.createdAt}, 'unixepoch')`;

  const [incomingDaily, incomingWeekly, incomingMonthly, conversationsDaily, conversationsWeekly, conversationsMonthly, contactsDaily, contactsWeekly, contactsMonthly] = await Promise.all([
    db
      .select({ period: msgDayExpr.as("period"), count: count() })
      .from(messages)
      .where(and(eq(messages.userId, uid), eq(messages.senderRole, "contact"), isNotNull(messages.createdAt)))
      .groupBy(msgDayExpr)
      .orderBy(asc(msgDayExpr)),
    db
      .select({ period: msgWeekExpr.as("period"), count: count() })
      .from(messages)
      .where(and(eq(messages.userId, uid), eq(messages.senderRole, "contact"), isNotNull(messages.createdAt)))
      .groupBy(msgWeekExpr)
      .orderBy(asc(msgWeekExpr)),
    db
      .select({ period: msgMonthExpr.as("period"), count: count() })
      .from(messages)
      .where(and(eq(messages.userId, uid), eq(messages.senderRole, "contact"), isNotNull(messages.createdAt)))
      .groupBy(msgMonthExpr)
      .orderBy(asc(msgMonthExpr)),

    db
      .select({ period: convDayExpr.as("period"), count: count() })
      .from(conversations)
      .where(and(eq(conversations.userId, uid), isNotNull(conversations.createdAt)))
      .groupBy(convDayExpr)
      .orderBy(asc(convDayExpr)),
    db
      .select({ period: convWeekExpr.as("period"), count: count() })
      .from(conversations)
      .where(and(eq(conversations.userId, uid), isNotNull(conversations.createdAt)))
      .groupBy(convWeekExpr)
      .orderBy(asc(convWeekExpr)),
    db
      .select({ period: convMonthExpr.as("period"), count: count() })
      .from(conversations)
      .where(and(eq(conversations.userId, uid), isNotNull(conversations.createdAt)))
      .groupBy(convMonthExpr)
      .orderBy(asc(convMonthExpr)),

    db
      .select({ period: contactDayExpr.as("period"), count: count() })
      .from(contacts)
      .where(and(eq(contacts.userId, uid), isNotNull(contacts.createdAt)))
      .groupBy(contactDayExpr)
      .orderBy(asc(contactDayExpr)),
    db
      .select({ period: contactWeekExpr.as("period"), count: count() })
      .from(contacts)
      .where(and(eq(contacts.userId, uid), isNotNull(contacts.createdAt)))
      .groupBy(contactWeekExpr)
      .orderBy(asc(contactWeekExpr)),
    db
      .select({ period: contactMonthExpr.as("period"), count: count() })
      .from(contacts)
      .where(and(eq(contacts.userId, uid), isNotNull(contacts.createdAt)))
      .groupBy(contactMonthExpr)
      .orderBy(asc(contactMonthExpr)),
  ]);

  const incomingMessages: TrendSeries = {
    daily: normalizeTrendRows(incomingDaily),
    weekly: normalizeTrendRows(incomingWeekly),
    monthly: normalizeTrendRows(incomingMonthly),
  };

  const newConversations: TrendSeries = {
    daily: normalizeTrendRows(conversationsDaily),
    weekly: normalizeTrendRows(conversationsWeekly),
    monthly: normalizeTrendRows(conversationsMonthly),
  };

  const newContacts: TrendSeries = {
    daily: normalizeTrendRows(contactsDaily),
    weekly: normalizeTrendRows(contactsWeekly),
    monthly: normalizeTrendRows(contactsMonthly),
  };

  // --- Performa chatbot ---
  const conversationsCountExpr = count(conversations.id);
  const topChatbotsByConversationsRaw = await db
    .select({
      chatbotId: chatbots.id,
      title: chatbots.title,
      conversations: conversationsCountExpr,
    })
    .from(chatbots)
    .innerJoin(conversations, eq(chatbots.id, conversations.chatbotId))
    .where(and(eq(chatbots.userId, uid), eq(conversations.userId, uid)))
    .groupBy(chatbots.id, chatbots.title)
    .orderBy(desc(conversationsCountExpr))
    .limit(5);

  const messagesCountExpr = count(messages.id);
  const topChatbotsByMessagesRaw = await db
    .select({
      chatbotId: chatbots.id,
      title: chatbots.title,
      messages: messagesCountExpr,
    })
    .from(chatbots)
    .innerJoin(conversations, eq(chatbots.id, conversations.chatbotId))
    .innerJoin(messages, eq(conversations.id, messages.conversationId))
    .where(and(eq(chatbots.userId, uid), eq(conversations.userId, uid), eq(messages.userId, uid)))
    .groupBy(chatbots.id, chatbots.title)
    .orderBy(desc(messagesCountExpr))
    .limit(5);

  const avgLengthExpr = sql<number>`avg(length(${messages.text}))`;
  const topChatbotsByAvgUserMessageLengthRaw = await db
    .select({
      chatbotId: chatbots.id,
      title: chatbots.title,
      avgLength: avgLengthExpr,
    })
    .from(chatbots)
    .innerJoin(conversations, eq(chatbots.id, conversations.chatbotId))
    .innerJoin(messages, eq(conversations.id, messages.conversationId))
    .where(and(eq(chatbots.userId, uid), eq(conversations.userId, uid), eq(messages.userId, uid), eq(messages.senderRole, "contact")))
    .groupBy(chatbots.id, chatbots.title)
    .orderBy(desc(avgLengthExpr))
    .limit(5);

  return c.json(
    {
      summary: {
        totalChatbots,
        totalConversations,
        totalMessages,
        totalContacts,
        totalAiResponses,
        avgMessagesPerConversation,
      },
      trends: {
        incomingMessages,
        newConversations,
        newContacts,
        autoReplyRatio,
      },
      performance: {
        topChatbotsByConversations: topChatbotsByConversationsRaw.map(row => ({
          chatbotId: row.chatbotId,
          title: row.title,
          conversations: Number(row.conversations ?? 0),
        })),
        topChatbotsByMessages: topChatbotsByMessagesRaw.map(row => ({
          chatbotId: row.chatbotId,
          title: row.title,
          messages: Number(row.messages ?? 0),
        })),
        topChatbotsByAvgUserMessageLength: topChatbotsByAvgUserMessageLengthRaw.map(row => ({
          chatbotId: row.chatbotId,
          title: row.title,
          avgLength: Number(row.avgLength ?? 0),
        })),
      },
    },
    HttpStatusCodes.OK,
  );
};
