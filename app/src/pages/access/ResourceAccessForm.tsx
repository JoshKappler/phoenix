import { css } from "@emotion/react";
import { useState } from "react";
import { graphql, useMutation } from "react-relay";

import {
  Alert,
  Button,
  Card,
  ComboBox,
  ComboBoxItem,
  Flex,
  Icon,
  Icons,
  Label,
  ListBox,
  Popover,
  Select,
  SelectChevronUpDownIcon,
  SelectItem,
  SelectValue,
  Text,
  View,
} from "@phoenix/components";
import { ConfirmButton } from "@phoenix/components/ConfirmButton";
import { tableCSS } from "@phoenix/components/table/styles";
import { useNotifySuccess } from "@phoenix/contexts";

import type { ResourceAccessFormGrantMutation } from "./__generated__/ResourceAccessFormGrantMutation.graphql";
import type { ResourceAccessFormRevokeMutation } from "./__generated__/ResourceAccessFormRevokeMutation.graphql";

const pageCSS = css`
  overflow-y: auto;
  padding: var(--global-dimension-size-400);
`;

const modalPageCSS = css`
  overflow-y: auto;
  padding: var(--global-dimension-size-200);
`;

const innerCSS = css`
  max-width: 800px;
  min-width: 500px;
  width: 100%;
  margin: 0 auto;
  box-sizing: border-box;
`;

const addAccessRowCSS = css`
  display: grid;
  grid-template-columns: minmax(260px, 1fr) minmax(180px, 220px) auto;
  gap: var(--global-dimension-size-200);
  align-items: start;

  @media (max-width: 700px) {
    grid-template-columns: 1fr;
  }
`;

const roleFieldCSS = css`
  min-width: 0;
`;

const grantButtonCSS = css`
  margin-top: var(--global-dimension-size-300);
`;

/**
 * Which resource a grant targets. Exactly one key is set (the GraphQL input is `@oneOf`),
 * which also selects the resource noun shown in the UI copy.
 */
export type AccessObjectInput =
  | { projectId: string }
  | { datasetId: string }
  | { promptId: string };

/**
 * A grant subject. Users and groups are identified by their Relay global id; "All users"
 * (everyone) carries no id. Mirrors the `@oneOf` GraphQL subject input.
 */
type SubjectInput =
  | { userId: string }
  | { userGroupId: string }
  | { isEveryone: true };

export type SubjectItem = { id: string; label: string };

type PermissionSet = {
  readonly id: string;
  readonly name: string;
  readonly permissions: readonly string[];
};

type AccessGrant = {
  readonly subjectKind: string;
  readonly subjectId: string | null;
  readonly subjectName: string;
  readonly roleId: string | null;
  readonly roleName: string;
};

const EVERYONE = "EVERYONE";

function encodeSubject(kind: "USER" | "GROUP", id: string): string {
  return `${kind}:${id}`;
}

/** Build the combobox options (everyone, then people, then groups) from query results. */
export function buildSubjectItems(
  users: ReadonlyArray<{
    readonly id: string;
    readonly username: string | null;
    readonly email: string | null;
  }>,
  userGroups: ReadonlyArray<{ readonly id: string; readonly name: string }>
): SubjectItem[] {
  const items: SubjectItem[] = [
    { id: EVERYONE, label: "All users · everyone" },
  ];
  for (const user of users) {
    items.push({
      id: encodeSubject("USER", user.id),
      label: `${user.email || user.username || "user"} · person`,
    });
  }
  for (const group of userGroups) {
    items.push({
      id: encodeSubject("GROUP", group.id),
      label: `${group.name} · group`,
    });
  }
  return items;
}

/** A plain-language summary of an object role, derived from the permissions it confers. */
function describeRole(permissions: readonly string[]): string {
  if (permissions.includes("MANAGE_ACCESS")) {
    return "Can view, edit, and manage who has access";
  }
  if (permissions.includes("EDIT")) {
    return "Can view and edit";
  }
  return "Can view";
}

function resolveSelectedSubject(selectedSubject: string): SubjectInput | null {
  if (selectedSubject === EVERYONE) {
    return { isEveryone: true };
  }
  const separatorIndex = selectedSubject.indexOf(":");
  if (separatorIndex < 0) {
    return null;
  }
  const kind = selectedSubject.slice(0, separatorIndex);
  const subjectId = selectedSubject.slice(separatorIndex + 1);
  if (kind === "USER") {
    return { userId: subjectId };
  }
  if (kind === "GROUP") {
    return { userGroupId: subjectId };
  }
  return null;
}

/** The `SubjectInput` that re-identifies an existing grant for revoke / role change. */
function grantSubject(grant: AccessGrant): SubjectInput {
  if (grant.subjectKind === "EVERYONE") {
    return { isEveryone: true };
  }
  if (grant.subjectKind === "GROUP") {
    return { userGroupId: grant.subjectId as string };
  }
  return { userId: grant.subjectId as string };
}

/**
 * The shared "manage access" surface for a project, dataset, or prompt. Renders the fail-closed
 * sharing model — admins always have access; everyone else is added by explicit grant — over the
 * unified access-grant mutations. The owning page supplies the resource, its current grants, the
 * grantable subjects/roles, and search + refetch hooks; the resource noun is taken from `object`.
 */
export function ResourceAccessForm({
  object,
  grants,
  permissionSets,
  subjectItems,
  onSubjectSearchChange,
  onRefresh,
  isModal = false,
}: {
  object: AccessObjectInput;
  grants: ReadonlyArray<AccessGrant>;
  permissionSets: ReadonlyArray<PermissionSet>;
  subjectItems: ReadonlyArray<SubjectItem>;
  onSubjectSearchChange: (value: string) => void;
  onRefresh: () => void;
  isModal?: boolean;
}) {
  const [error, setError] = useState<string | null>(null);
  const [selectedSubject, setSelectedSubject] = useState<string | null>(null);
  const [selectedRoleId, setSelectedRoleId] = useState<string | null>(null);
  const notifySuccess = useNotifySuccess();

  const [commitGrant, isGranting] =
    useMutation<ResourceAccessFormGrantMutation>(graphql`
      mutation ResourceAccessFormGrantMutation($input: AccessGrantInput!) {
        grantAccess(input: $input) {
          __typename
        }
      }
    `);

  const [commitRevoke, isRevoking] =
    useMutation<ResourceAccessFormRevokeMutation>(graphql`
      mutation ResourceAccessFormRevokeMutation($input: AccessGrantInput!) {
        revokeAccess(input: $input) {
          __typename
        }
      }
    `);

  const noun =
    "projectId" in object
      ? "project"
      : "datasetId" in object
        ? "dataset"
        : "prompt";
  const isMutating = isGranting || isRevoking;
  const selectedRole = permissionSets.find((role) => role.id === selectedRoleId);
  const selectedRoleDescription = selectedRole
    ? describeRole(selectedRole.permissions)
    : "Defaults to Viewer — can view";

  const grantAccess = () => {
    if (!selectedSubject) return;
    const subject = resolveSelectedSubject(selectedSubject);
    if (subject == null) {
      setError("Could not resolve the selected subject.");
      return;
    }
    setError(null);
    commitGrant({
      variables: {
        input: { object, subject, permissionSetId: selectedRoleId || null },
      },
      onCompleted: () => {
        notifySuccess({
          title: "Access granted",
          message: `The subject can now access this ${noun}.`,
        });
        setSelectedSubject(null);
        setSelectedRoleId(null);
        onRefresh();
      },
      onError: (e) => setError(e.message),
    });
  };

  const revokeAccess = (subject: SubjectInput) => {
    setError(null);
    commitRevoke({
      variables: { input: { object, subject } },
      onCompleted: () => {
        notifySuccess({
          title: "Access revoked",
          message: "Access has been removed.",
        });
        onRefresh();
      },
      onError: (e) => setError(e.message),
    });
  };

  const changeRole = (subject: SubjectInput, roleId: string) => {
    setError(null);
    commitGrant({
      variables: { input: { object, subject, permissionSetId: roleId } },
      onCompleted: () => {
        notifySuccess({
          title: "Access updated",
          message: "The access level has been changed.",
        });
        onRefresh();
      },
      onError: (e) => setError(e.message),
    });
  };

  return (
    <div css={isModal ? modalPageCSS : pageCSS}>
      <div css={innerCSS}>
        <Flex direction="column" gap="size-200">
          {error && <Alert variant="danger">{error}</Alert>}
          <Alert variant="info" banner>
            A {noun} is visible to administrators only until it is shared. Grant
            people or groups below to give them access; removing every grant
            returns it to administrators only.
          </Alert>
          <Card title="Add access" titleSeparator={false}>
            <View padding="size-200">
              <Flex direction="column" gap="size-200">
                <Text color="text-700">
                  Choose who should get access, then select the level of access
                  they should receive.
                </Text>
                <div css={addAccessRowCSS}>
                  <ComboBox
                    label="Person or group"
                    placeholder="Search people and groups…"
                    width="100%"
                    selectedKey={selectedSubject}
                    onSelectionChange={(key) =>
                      setSelectedSubject(key == null ? null : String(key))
                    }
                    onInputChange={onSubjectSearchChange}
                    renderEmptyState={() => "No people or groups found"}
                  >
                    {subjectItems.map((item) => (
                      <ComboBoxItem
                        key={item.id}
                        id={item.id}
                        textValue={item.label}
                      >
                        {item.label}
                      </ComboBoxItem>
                    ))}
                  </ComboBox>
                  <Flex direction="column" gap="size-50" css={roleFieldCSS}>
                    <Select
                      value={selectedRoleId ?? undefined}
                      onChange={(key) =>
                        setSelectedRoleId(key == null ? null : String(key))
                      }
                      placeholder="Viewer"
                    >
                      <Label>Access level</Label>
                      <Button>
                        <SelectValue />
                        <SelectChevronUpDownIcon />
                      </Button>
                      <Popover>
                        <ListBox>
                          {permissionSets.map((role) => (
                            <SelectItem key={role.id} id={role.id}>
                              {role.name}
                            </SelectItem>
                          ))}
                        </ListBox>
                      </Popover>
                    </Select>
                    <Text size="XS" color="text-700">
                      {selectedRoleDescription}
                    </Text>
                  </Flex>
                  <Button
                    size="S"
                    variant="primary"
                    isDisabled={!selectedSubject || isMutating}
                    leadingVisual={<Icon svg={<Icons.PlusCircleOutline />} />}
                    onPress={grantAccess}
                    css={grantButtonCSS}
                  >
                    Grant access
                  </Button>
                </div>
              </Flex>
            </View>
          </Card>
          <Card title={`Who can access this ${noun}`} titleSeparator={false}>
            <table css={tableCSS}>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Type</th>
                  <th>Access level</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    <Flex direction="row" gap="size-100" alignItems="center">
                      <Icon svg={<Icons.ShieldOutline />} />
                      Administrators
                    </Flex>
                  </td>
                  <td>Role</td>
                  <td>Full access</td>
                  <td>
                    <Flex justifyContent="end">
                      <Text size="XS" color="text-700">
                        always
                      </Text>
                    </Flex>
                  </td>
                </tr>
                {grants.length === 0 ? (
                  <tr>
                    <td colSpan={4}>
                      <View padding="size-200">
                        <Text color="text-700">
                          No one else has been granted access — only
                          administrators can see this {noun}.
                        </Text>
                      </View>
                    </td>
                  </tr>
                ) : (
                  grants.map((grant) => {
                    const subject = grantSubject(grant);
                    return (
                      <tr key={`${grant.subjectKind}:${grant.subjectId}`}>
                        <td>{grant.subjectName}</td>
                        <td>
                          {grant.subjectKind === "USER"
                            ? "Person"
                            : grant.subjectKind === "GROUP"
                              ? "Group"
                              : "Everyone"}
                        </td>
                        <td>
                          <Select
                            value={grant.roleId ?? undefined}
                            onChange={(key) => {
                              if (key == null) return;
                              changeRole(subject, String(key));
                            }}
                          >
                            <Label>Access level</Label>
                            <Button size="S">
                              <SelectValue>{grant.roleName}</SelectValue>
                              <SelectChevronUpDownIcon />
                            </Button>
                            <Popover>
                              <ListBox>
                                {permissionSets.map((role) => (
                                  <SelectItem key={role.id} id={role.id}>
                                    {role.name}
                                  </SelectItem>
                                ))}
                              </ListBox>
                            </Popover>
                          </Select>
                        </td>
                        <td>
                          <Flex justifyContent="end">
                            <ConfirmButton
                              buttonText="Remove"
                              title="Remove access"
                              message={`Remove ${grant.subjectName}'s access to this ${noun}?`}
                              confirmText="Remove access"
                              isDisabled={isMutating}
                              onConfirm={() => revokeAccess(subject)}
                            />
                          </Flex>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </Card>
        </Flex>
      </div>
    </div>
  );
}
