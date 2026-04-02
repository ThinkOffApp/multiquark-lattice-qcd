import type { ApprovalEntity, ApprovalStatus, UserApprovalSnapshot, UserRole } from "@/lib/types";
import { createSupabaseServerClient } from "@/lib/supabase/server";
import { createSupabaseAdminClient } from "@/lib/supabase/admin";
import { isSupabaseConfigured, isSupabaseAdminConfigured } from "@/lib/supabase/config";

const mockAssociations: ApprovalEntity[] = [
  {
    id: "assoc-1",
    name: "Finnish Beauceron Club",
    status: "pending",
    contactEmail: "board@beauceron.fi",
    meta: "Working dogs",
    createdAt: "2026-03-21T10:00:00.000Z",
  },
  {
    id: "assoc-2",
    name: "Spitz Rescue Network",
    status: "active",
    contactEmail: "ops@spitzrescue.fi",
    meta: "Nordic breeds",
    createdAt: "2026-03-18T08:00:00.000Z",
  },
];

const mockVendors: ApprovalEntity[] = [
  {
    id: "vendor-1",
    name: "NordFeed Oy",
    status: "pending",
    contactEmail: "sales@nordfeed.fi",
    meta: "Finland",
    createdAt: "2026-03-20T11:00:00.000Z",
  },
  {
    id: "vendor-2",
    name: "Hunt & Hound Nutrition",
    status: "active",
    contactEmail: "partner@huntandhound.eu",
    meta: "Germany",
    createdAt: "2026-03-12T14:00:00.000Z",
  },
];

function formatApprovalEntity(
  id: string,
  name: string,
  status: ApprovalStatus,
  contactEmail: string | null,
  meta: string | null,
  createdAt: string | null,
) {
  return {
    id,
    name,
    status,
    contactEmail,
    meta,
    createdAt: createdAt ?? new Date(0).toISOString(),
  } satisfies ApprovalEntity;
}

async function getUserApprovalSnapshotWithClient(): Promise<UserApprovalSnapshot | null> {
  if (!isSupabaseConfigured()) {
    return null;
  }

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return null;
  }

  const { data: profile } = await supabase
    .from("user_profiles")
    .select("role, full_name")
    .eq("id", user.id)
    .maybeSingle();

  if (!profile) {
    return {
      role: null,
      status: null,
      organizationName: null,
      fullName: user.user_metadata?.full_name ?? null,
      email: user.email ?? null,
    };
  }

  if (profile.role === "association") {
    const { data: association } = await supabase
      .from("associations")
      .select("name, status")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false })
      .limit(1)
      .maybeSingle();

    return {
      role: profile.role,
      status: association?.status ?? null,
      organizationName: association?.name ?? null,
      fullName: profile.full_name ?? null,
      email: user.email ?? null,
    };
  }

  if (profile.role === "vendor") {
    const { data: vendor } = await supabase
      .from("vendors")
      .select("company_name, status")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false })
      .limit(1)
      .maybeSingle();

    return {
      role: profile.role,
      status: vendor?.status ?? null,
      organizationName: vendor?.company_name ?? null,
      fullName: profile.full_name ?? null,
      email: user.email ?? null,
    };
  }

  return {
    role: profile.role,
    status: "active",
    organizationName: "Admin workspace",
    fullName: profile.full_name ?? null,
    email: user.email ?? null,
  };
}

export async function getCurrentUserApprovalSnapshot() {
  return getUserApprovalSnapshotWithClient();
}

export async function getPostLoginDestination(userId: string) {
  if (!isSupabaseConfigured()) {
    return "/pending?notice=Supabase+is+not+configured+yet";
  }

  const supabase = await createSupabaseServerClient();
  const { data: profile } = await supabase
    .from("user_profiles")
    .select("role")
    .eq("id", userId)
    .maybeSingle();

  if (!profile) {
    return "/pending?error=Profile+setup+is+still+missing";
  }

  if (profile.role === "admin") {
    return "/admin";
  }

  if (profile.role === "association") {
    const { data: association } = await supabase
      .from("associations")
      .select("status")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(1)
      .maybeSingle();

    return association?.status === "active" ? "/association" : "/pending";
  }

  const { data: vendor } = await supabase
    .from("vendors")
    .select("status")
    .eq("user_id", userId)
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  return vendor?.status === "active" ? "/vendor" : "/pending";
}

export async function getAssociationApprovals() {
  if (!isSupabaseAdminConfigured()) {
    return mockAssociations;
  }

  try {
    const supabase = createSupabaseAdminClient();
    const { data, error } = await supabase
      .from("associations")
      .select("id, name, status, contact_email, focus_animal, created_at")
      .order("created_at", { ascending: false });

    if (error || !data) {
      return mockAssociations;
    }

    return data.map((row) =>
      formatApprovalEntity(
        row.id,
        row.name,
        row.status,
        row.contact_email,
        row.focus_animal,
        row.created_at,
      ),
    );
  } catch {
    return mockAssociations;
  }
}

export async function getVendorApprovals() {
  if (!isSupabaseAdminConfigured()) {
    return mockVendors;
  }

  try {
    const supabase = createSupabaseAdminClient();
    const { data, error } = await supabase
      .from("vendors")
      .select("id, company_name, status, contact_email, country, created_at")
      .order("created_at", { ascending: false });

    if (error || !data) {
      return mockVendors;
    }

    return data.map((row) =>
      formatApprovalEntity(
        row.id,
        row.company_name,
        row.status,
        row.contact_email,
        row.country,
        row.created_at,
      ),
    );
  } catch {
    return mockVendors;
  }
}

export function getStatusTone(status: ApprovalStatus | null) {
  if (status === "active") return "success";
  if (status === "rejected" || status === "suspended") return "error";
  return "pending";
}

export function getRoleLabel(role: UserRole | null) {
  if (role === "association") return "Association";
  if (role === "vendor") return "Producer";
  if (role === "admin") return "Admin";
  return "Pending";
}
