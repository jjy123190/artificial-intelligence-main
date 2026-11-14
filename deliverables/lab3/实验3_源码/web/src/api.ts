export class ApiError extends Error {
  constructor(
    message: string,
    public code: string,
    public status: number,
  ) {
    super(message);
  }
}
export type RequestOptions = RequestInit & { timeoutMs?: number };

export async function request<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { timeoutMs = 30_000, signal, ...init } = options;
  const controller = new AbortController();
  let timedOut = false;
  const abort = () => controller.abort(signal?.reason);
  if (signal?.aborted) abort();
  else signal?.addEventListener("abort", abort, { once: true });
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    const headers = new Headers(init.headers);
    if (init.body && !headers.has("Content-Type"))
      headers.set("Content-Type", "application/json");
    let response: Response;
    try {
      response = await fetch(path, {
        ...init,
        headers,
        signal: controller.signal,
      });
    } catch (error) {
      if (controller.signal.aborted) throw error;
      throw new ApiError(
        "无法连接本地引擎，请确认网站服务仍在运行。",
        "network",
        0,
      );
    }
    let data: unknown;
    try {
      data = await response.json();
    } catch (error) {
      if (controller.signal.aborted) throw error;
      throw new ApiError(
        "服务器没有返回有效数据，请重试。",
        "invalid_response",
        response.status,
      );
    }
    // A response may finish parsing after its caller has left the page.
    if (controller.signal.aborted) throw controller.signal.reason;
    if (!response.ok) {
      const value =
        data && typeof data === "object"
          ? (data as { error?: { message?: unknown; code?: unknown } }).error
          : undefined;
      throw new ApiError(
        typeof value?.message === "string"
          ? value.message
          : "操作未完成，请重试。",
        typeof value?.code === "string" ? value.code : "request_failed",
        response.status,
      );
    }
    return data as T;
  } catch (error) {
    if (signal?.aborted)
      throw signal.reason ?? new DOMException("请求已取消", "AbortError");
    if (timedOut)
      throw new ApiError(
        "等待结果超时。可以重试，或先检查本地引擎是否正常运行。",
        "timeout",
        408,
      );
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
  }
}
export function canonical(value: unknown): string {
  const normalize = (item: unknown): unknown =>
    Array.isArray(item)
      ? item.map(normalize)
      : item && typeof item === "object"
        ? Object.fromEntries(
            Object.entries(item)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([k, v]) => [k, normalize(v)]),
          )
        : item;
  return JSON.stringify(normalize(value));
}
export function downloadJson(value: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], {
      type: "application/json;charset=utf-8",
    }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export const fmt = (value: number, digits = 3) =>
  value.toLocaleString("zh-CN", { maximumFractionDigits: digits });
export function shortName(name: string): string {
  return name
    .replace(/股份有限公司|有限责任公司|有限公司|合伙企业|股份公司/g, "")
    .replace(/集团$/, "");
}
