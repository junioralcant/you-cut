---
name: executar-qa
description: Validates feature implementation against PRD, Tech Spec, and Tasks through E2E testing with Maestro, accessibility verification (WCAG 2.2), and visual analysis. Documents all bugs found with screenshot evidence and generates a comprehensive QA report. Use when the user asks to run QA, validate a feature, or test implementation completeness. Do not use for code review, bug fixing, or task implementation.
---

# QA Execution

## Procedures

**Step 1: Documentation Analysis (Mandatory)**

1. Read the PRD at `./tasks/prd-[feature-slug]/prd.md` and extract ALL numbered functional requirements.
2. Read the Tech Spec at `./tasks/prd-[feature-slug]/techspec.md` and verify implemented technical decisions.
3. Read Tasks at `./tasks/prd-[feature-slug]/tasks.md` and verify completion status of each task.
4. Create a verification checklist based on the requirements.
5. Do NOT skip this step — understanding requirements is fundamental for QA.

**Step 2: Environment Preparation (Mandatory)**

1. Verify the application is running on localhost.
2. Use Maestro to start or connect to a device/emulator when needed.
3. Launch the target application with Maestro and confirm the initial screen loaded correctly.

**Step 3: E2E Tests with Maestro (Mandatory)**

1. Read the available Maestro tool reference before executing flows.
2. For each functional requirement from the PRD:
   a. Navigate to the feature.
   b. Execute the expected flow.
   c. Verify the result.
   d. Capture screenshot evidence.
   e. Mark as PASSED or FAILED.
3. Always inspect the current UI hierarchy before interacting so selectors are based on the actual rendered state.
4. Prefer Maestro commands and flows such as app launch, tap, input, scroll, screenshots, and assertions to validate behavior.
5. If a reusable flow file already exists, run it; otherwise execute focused ad hoc Maestro commands to validate the requirement.

**Step 4: Accessibility Verification (Mandatory)**

1. Verify for each screen/component:
   - Keyboard navigation works (Tab, Enter, Escape).
   - Interactive elements have descriptive labels.
   - Images have appropriate alt text.
   - Color contrast is adequate.
   - Forms have labels associated to inputs.
   - Error messages are clear and accessible.
2. Use Maestro interactions to test focus movement, keyboard actions when supported, and dismiss flows such as Escape or Back.
3. Use screenshots and view hierarchy inspection to verify labels, text alternatives, and semantic structure exposed by the UI.
4. Follow WCAG 2.2 standard.

**Step 5: Visual Verification (Mandatory)**

1. Capture screenshots of main screens with Maestro.
2. Verify layouts in different states (empty, with data, error).
3. Document visual inconsistencies found.
4. Verify responsiveness if applicable.

**Step 6: Bug Documentation**

1. For each bug found, document with:
   - Bug ID, Description, Severity (High/Medium/Low), Screenshot.
2. Save bugs to `./tasks/prd-[feature-slug]/bugs.md`.
3. If a blocking bug is found, document and report immediately.

**Step 7: Generate QA Report (Mandatory)**

1. Read the report template at `assets/qa-report-template.md`.
2. Fill in all sections with actual results.
3. Set status to APPROVED only when ALL PRD requirements are verified and functioning.

## Error Handling

- If the application is not running, instruct the user to start it with `npm run dev` before retrying.
- If Maestro tooling is unavailable, report the error and suggest running E2E tests manually with `npm run e2e:maestro` or the project's equivalent Node-based script.
- If a blocking bug prevents testing subsequent features, document it and continue with testable areas.
