-- Central catalogue and release registry (PostgreSQL). Idempotent via schema_migrations.
CREATE TABLE IF NOT EXISTS catalogue_revision (
    revision        text PRIMARY KEY,
    catalogue       text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    parent_revision text REFERENCES catalogue_revision(revision),
    manifest        jsonb NOT NULL,
    manifest_sha256 text NOT NULL CHECK (length(manifest_sha256) = 64),
    approved_by     text,
    approved_at     timestamptz
);

CREATE TABLE IF NOT EXISTS product (
    sku               text NOT NULL,
    revision          text NOT NULL REFERENCES catalogue_revision(revision) ON DELETE RESTRICT,
    family            text NOT NULL,
    artwork_revision  text NOT NULL,
    quantity          text,
    variant           text,
    approval_state    text NOT NULL CHECK (approval_state IN ('draft','approved','fixture','retired')),
    usage_rights      text NOT NULL,
    provenance        text NOT NULL,
    PRIMARY KEY (revision, sku)
);

-- A verified GTIN maps to exactly one SKU inside a catalogue revision.
CREATE TABLE IF NOT EXISTS gtin_mapping (
    revision  text NOT NULL,
    gtin14    char(14) NOT NULL CHECK (gtin14 ~ '^[0-9]{14}$'),
    sku       text NOT NULL,
    verified_source text NOT NULL,
    PRIMARY KEY (revision, gtin14),
    FOREIGN KEY (revision, sku) REFERENCES product(revision, sku) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS gtin_mapping_sku ON gtin_mapping(revision, sku);

CREATE TABLE IF NOT EXISTS reference_image (
    revision  text NOT NULL,
    sku       text NOT NULL,
    side      text NOT NULL CHECK (side IN ('front','back','edge')),
    uri       text NOT NULL,
    sha256    char(64) NOT NULL,
    landmarks jsonb,
    PRIMARY KEY (revision, sku, sha256),
    FOREIGN KEY (revision, sku) REFERENCES product(revision, sku) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS release_bundle (
    bundle_id          text PRIMARY KEY,
    created_at         timestamptz NOT NULL DEFAULT now(),
    schema_version     text NOT NULL,
    catalogue_revision text NOT NULL REFERENCES catalogue_revision(revision),
    thresholds_version text NOT NULL,
    manifest           jsonb NOT NULL,
    signature_b64      text NOT NULL,
    state              text NOT NULL DEFAULT 'candidate' CHECK (state IN ('candidate','shadow','canary','promoted','revoked'))
);

CREATE TABLE IF NOT EXISTS station_deployment (
    station_id   text NOT NULL,
    bundle_id    text NOT NULL REFERENCES release_bundle(bundle_id),
    activated_at timestamptz NOT NULL DEFAULT now(),
    previous_bundle_id text REFERENCES release_bundle(bundle_id),
    PRIMARY KEY (station_id, activated_at)
);
CREATE INDEX IF NOT EXISTS station_deployment_latest ON station_deployment(station_id, activated_at DESC);

-- Idempotent sink for station events (consumer-side dedup by event_id).
CREATE TABLE IF NOT EXISTS station_event (
    event_id    char(32) PRIMARY KEY,
    station_id  text NOT NULL,
    camera      text NOT NULL,
    epoch       integer NOT NULL,
    track       text NOT NULL,
    seq         integer NOT NULL CHECK (seq > 0),
    kind        text NOT NULL,
    state       jsonb NOT NULL,
    captured_at timestamptz NOT NULL,
    received_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (station_id, camera, epoch, track, seq)
);
