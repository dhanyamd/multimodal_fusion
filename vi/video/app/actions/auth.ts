/* eslint-disable @typescript-eslint/no-unused-vars */

"use server";

import { hash } from "bcryptjs";
import { db } from "~/server/db";
import crypto from "crypto";
import { SignupSchema, signupSchema } from "@/schema/auth";

export async function registerUser(data: SignupSchema) {
  try {
    // Server-side validation
    const result = signupSchema.safeParse(data);
    if (!result.success) {
      return { error: "Invalid data" };
    }

    const { name, email, password } = data;

    // Check if user exist
    const existingUser = await db.user.findUnique({
      where: { email },
    });

    if (existingUser) {
      return { error: "User already exist" };
    }

    const hashedPassword = await hash(password, 12);

    await db.user.create({
      data: {
        name,
        email,
        password: hashedPassword,
        apiQuotas: {
          create: {
            secretKey: `sa_live_${crypto.randomBytes(24).toString("hex")}`,
            resetDate: new Date(),
          },
        },
      },
    });

    return { success: true };
  } catch (error) {
    return { error: "Something went wrong" };
  }
}