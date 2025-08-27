# --- Auto-detect project, scheme, bundle id, device ---
PROJECT := $(shell ls -1 *.xcodeproj 2>/dev/null | head -1)
ifeq ($(strip $(PROJECT)),)
$(warning No .xcodeproj found in this folder.)
$(warning Create an Xcode visionOS App project here (File → New → App → Platform: visionOS), then rerun `make run`.)
endif

# First scheme listed in the project (usually the app)
SCHEME ?= $(shell xcodebuild -list -project "$(PROJECT)" 2>/dev/null | awk '/Schemes:/{f=1;next} f && NF {print; exit}')

# Fallback: use project name without extension if scheme missing
ifeq ($(strip $(SCHEME)),)
SCHEME := $(basename $(PROJECT))
endif

# Use scheme name as app name by default
APP_NAME ?= $(SCHEME)

# Get bundle id from the first PRODUCT_BUNDLE_IDENTIFIER in the pbxproj
BUNDLE_ID ?= $(shell grep -Eo 'PRODUCT_BUNDLE_IDENTIFIER = [^;]+' "$(PROJECT)/project.pbxproj" 2>/dev/null | head -1 | awk '{print $$3}')

# If still empty, use a safe default (overridden at build time)
ifeq ($(strip $(BUNDLE_ID)),)
BUNDLE_ID := com.example.$(APP_NAME)
endif

# Pick the first available visionOS device name
SIM_DEVICE_NAME ?= $(shell xcrun simctl list devices 2>/dev/null | sed -n '/visionOS/,$$p' | awk -F'[()]' '/Apple Vision Pro/ && $$0 !~ /unavailable/ {gsub(/^[ -]+|[ -]+$$/,"",$$1); print $$1; exit}')

# If not found, keep a reasonable default (may exist on most dev machines)
ifeq ($(strip $(SIM_DEVICE_NAME)),)
SIM_DEVICE_NAME := Apple Vision Pro (1st generation)
endif

PLATFORM ?= visionOS
CONFIG   ?= Debug

define GET_UDID
xcrun simctl list devices | awk -v name="$(SIM_DEVICE_NAME)" 'index($$0,name){ if (match($$0, /\(([A-F0-9-]{36})\)/, m)) { print m[1]; exit } }'
endef

.PHONY: info
info:
	@echo "Project:      $(PROJECT)"
	@echo "Scheme:       $(SCHEME)"
	@echo "App Name:     $(APP_NAME)"
	@echo "Bundle ID:    $(BUNDLE_ID)"
	@echo "Simulator:    $(SIM_DEVICE_NAME)"

.PHONY: list-devices
list-devices:
	@echo "=== visionOS devices ==="
	@xcrun simctl list devices | sed -n '/visionOS/,$$p' || true

.PHONY: boot
boot:
	@UDID="$$( $(GET_UDID) )"; \
	if [ -z "$$UDID" ]; then \
	  echo "Simulator device '$(SIM_DEVICE_NAME)' not found."; \
	  echo "Use 'make list-devices' and then re-run with SIM_DEVICE_NAME=\"Exact Name\" make run"; \
	  exit 1; \
	fi; \
	echo "Booting $$UDID …"; \
	xcrun simctl bootstatus $$UDID -b || true; \
	open -a Simulator || true

.PHONY: build
build:
	@echo "Building $(SCHEME) for $(PLATFORM) Simulator…"
	@xcodebuild \
	  -project "$(PROJECT)" \
	  -scheme "$(SCHEME)" \
	  -configuration "$(CONFIG)" \
	  -destination 'platform=$(PLATFORM) Simulator,name=$(SIM_DEVICE_NAME)' \
	  -derivedDataPath .derived \
	  PRODUCT_BUNDLE_IDENTIFIER="$(BUNDLE_ID)" \
	  clean build

.PHONY: install
install:
	@UDID="$$( $(GET_UDID) )"; \
	APP=$$(/usr/libexec/PlistBuddy -c "Print :ProductsPath" .derived/Info.plist 2>/dev/null)/"$(CONFIG)"/"$(APP_NAME)".app; \
	if [ ! -d "$$APP" ]; then \
	  echo "Built app not found at $$APP. Run 'make build' first."; \
	  exit 1; \
	fi; \
	echo "Installing $$APP on $$UDID …"; \
	xcrun simctl install $$UDID "$$APP"

.PHONY: launch
launch:
	@UDID="$$( $(GET_UDID) )"; \
	echo "Launching $(BUNDLE_ID) on $$UDID …"; \
	xcrun simctl launch $$UDID "$(BUNDLE_ID)" || true

.PHONY: run
run: info boot build install launch
	@echo "✅ Launched. Use 'make logs' to tail output."

.PHONY: logs
logs:
	@UDID="$$( $(GET_UDID) )"; \
	echo "Tailing logs for $(BUNDLE_ID) … (Ctrl+C to stop)"; \
	xcrun simctl spawn $$UDID log stream --predicate 'processImagePath CONTAINS[c] "$(APP_NAME)" || subsystem CONTAINS[c] "$(BUNDLE_ID)"'
