# Audit of AEC Agent Sources and Recommendations

The provided sources outline a sophisticated **"Local Sidecar" architecture** designed to bridge modern AI reasoning with legacy, single-threaded CAD environments like AutoCAD and Revit. The central findings of the audit are as follows:

*   **Architectural Strategy:** The system must use a four-layer stack—Frontend (Chainlit), Middleware (Python MCP), Bridge (.NET Sidecar), and Execution (CAD Kernel).
*   **Protocol Choice:** Standard `stdio` transport is rejected in favour of **SSE (Server-Sent Events) over HTTP** to ensure stability in multi-user RDP environments.
*   **Data Philosophy:** The AI should not use complex "God Tools"; instead, it should use **"Atomic Tools"** (granular functions) and **"Resources"** (active document state via URIs) to plan its own operations.
*   **The Threading Constraint:** Both AutoCAD and Revit are **single-threaded (STA)**. Requests from the asynchronous AI must be "marshalled" to the main thread using specific frameworks like `ExternalEvent` (Revit) or `Application.Idle` (AutoCAD).
*   **Scalability & Costs:** For 50 users, the audit recommends **AWS G4dn instances** and a hybrid financial model using **Autodesk Flex Tokens** for light users to save on annual subscriptions.

## Implementation Phases Overview

| Phase | Document | Description |
|-------|----------|-------------|
| 0 | `00_version_matrix.md` | Software compatibility requirements |
| 1 | `01_infrastructure.md` | AWS setup, port orchestration, I/O optimization |
| 2 | `02_autocad_sidecar.md` | .NET plugin with thread marshaling |
| 3 | `03_revit_sidecar.md` | pyRevit Routes API integration |
| 4 | `04_mcp_middleware.md` | Python MCP server with concurrency control |
| 5 | `05_frontend.md` | Chainlit UI with Steps visualization |
| 6 | `06_governance.md` | Logging, monitoring, alerting, cost optimization |
| 7 | `07_error_handling.md` | Error codes, recovery, rollback strategies |
| 8 | `08_testing_and_validation.md` | Unit, integration, E2E, and load testing |

## Core Recommendations Implemented in this Guide

1.  **Transport:** Use **SSE (Server-Sent Events)** over HTTP for robustness in RDP.
2.  **Infrastructure:** Isolate I/O by redirecting `%TEMP%` to NVMe storage.
3.  **Security:** Implement **Dynamic Port Orchestration** and **Session Token Validation** to prevent cross-user attacks on the shared Windows Server.
4.  **Performance:** Use **SQLite Caching** for Revit read-operations to achieve <50ms latency.
5.  **Reliability:** Implement circuit breakers, retry logic, and proper error handling (Phase 7).
6.  **Observability:** Centralized JSON logging, CloudWatch metrics, and alerting rules (Phase 6).
7.  **Quality:** Comprehensive testing pyramid with unit, integration, and E2E tests (Phase 8).
8.  **Compatibility:** Version matrix documenting supported AutoCAD/Revit/pyRevit versions (Phase 0).

## Quick Start Checklist

Before beginning implementation, ensure:

- [ ] Review `00_version_matrix.md` for software requirements
- [ ] Provision AWS infrastructure per `01_infrastructure.md`
- [ ] Decide on AutoCAD vs Revit priority (or parallel development)
- [ ] Set up logging infrastructure per `06_governance.md`
- [ ] Prepare test environment per `08_testing_and_validation.md`
