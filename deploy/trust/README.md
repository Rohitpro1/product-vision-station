Operator-provisioned Ed25519 **public** keys (`*.pub`) trusted for bundle verification.
Private signing keys are never stored in this repository. Generate a local test pair with
`pvs bundle keygen --key ~/.config/pvs/test.key --pub deploy/trust/local-test.pub` (the `.pub`
for local tests is git-ignored); production keys are created and held by the release operator.
