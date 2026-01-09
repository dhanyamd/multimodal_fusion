import { db } from "~/server/db";

export async function checkAndUpdateQuota(
  userId: string,
  deductFromQuota: boolean = true,
): Promise<boolean> {
  const quota = await db.apiQuota.findFirstOrThrow({
    where: { userId },
  });

  const now = new Date();
  const lastReset = new Date(quota.resetDate);
  const daysSinceLastReset =
    (now.getTime() - lastReset.getTime()) / (1000 * 60 * 60 * 24);

  if (daysSinceLastReset >= 30) {
    if (deductFromQuota) {
      await db.apiQuota.update({
        where: { id: quota.id },
        data: {
          resetDate: now,
          currentUsage: 1,
        },
      });
    }
    return true;
  }

  // Check if quota is exceeded
  if (quota.currentUsage >= quota.monthlyLimit) {
    return false;
  }

  if (deductFromQuota) {
    await db.apiQuota.update({
      where: { id: quota.id },
      data: {
        currentUsage: quota.currentUsage + 1,
      },
    });
  }

  return true;
}