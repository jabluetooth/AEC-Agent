# Prompt for Claude Code: AutoCAD PDF Rasterization Pipeline Architecture

## Task

Create a comprehensive markdown (.md) file that documents a NEW proposed approach for converting PDF technical drawings into AutoCAD vector components, and compare it against the existing/current pipeline. DO NOT write any code yet - only create the markdown documentation file.

## Context

### New Proposed Approach (What We Want to Build Next)

A completely different **rasterization orchestration** approach that will replace or augment the current system:

**Key Components:**

- **Groq LLM**: Acts as the messenger/chatbox only (user interface layer)
- **Gemini LLM**: Acts as the brain/orchestrator (decision-making layer)
- **AutoCAD Raster Tools**: The actual execution layer

**Gemini's Orchestration Responsibilities:**

1. Analyze the PDF file
2. Decide which AutoCAD raster tools to use for each element
3. Determine the sequence of tool execution
4. Make intelligent decisions about:
    - **Line types**: Identify patterns (solid, dashed, dotted, hidden, center lines)
    - **Layers**: Categorize and assign elements to appropriate layers
    - **Text understanding**: Understand what text SHOULD say based on context (NOT just OCR character recognition - semantic understanding)
    - **Line connectivity**: Ensure all lines are properly connected at intersections
    - **Symbols**: Recognize symbols and match them from database; if not found, search the internet
    - **Primitives**: Identify basic shapes (circles, rectangles, arcs, polylines)
    - **Followers**: Trace complex lines and curves

**AutoCAD Raster Tools to be Used:**

- **Followers**: Trace existing lines and curves from PDF
- **Primitives**: Extract and convert basic geometric shapes
- **Text** (understanding): Gemini determines what text should say based on drawing context
- **OCR**: Supplementary tool for character verification (separate from text understanding)
- **Line Type Detection**: Analyze and apply correct line types
- **Layer Assignment**: Intelligent categorization and organization
- **Symbol Recognition**: Match against database or internet search

**Critical Requirements:**

1. **Text Understanding**: Gemini must understand text semantically, not just recognize characters. For example, if a dimension shows a 100mm gap, the text should read "100" based on context understanding.
2. **Line Connectivity**: All lines must be validated to ensure they connect properly - no gaps, proper intersections.
3. **Symbols**: First check provided database, fallback to internet search if not present.

## Required Output

Create a markdown file that includes:

### 1. Executive Summary

- Overview of the current (existing) implementation
- Overview of the new proposed approach
- Clear statement of which approach is superior and why
- Recommendation on whether to replace or augment the current system

### 2. Current System Analysis

- Document how the existing pipeline works
- Identify its strengths
- Identify its limitations
- Explain what it does well vs. what could be improved

### 3. Detailed Comparison

- Side-by-side comparison table of both approaches
- Highlight key differences in methodology
- Explain the paradigm shift from "generative reconstruction" to "orchestrated rasterization"
- Show what changes in the data flow
- Illustrate differences in accuracy, fidelity, and output quality

### 4. New Proposed Architecture Documentation

- System architecture diagram (in markdown/text format)
- Component breakdown:
    - Groq's role (messenger)
    - Gemini's role (brain/orchestrator)
    - AutoCAD Raster Tools (execution layer)
    - Symbol Database
    - Internet search fallback

### 5. Detailed Workflow

- Step-by-step processing pipeline
- Decision-making logic for Gemini
- Tool selection criteria
- Execution sequence planning

### 6. Tool-by-Tool Breakdown

For each AutoCAD raster tool, document:

- What it does
- When Gemini should use it
- How it differs from generative approach
- Integration points

Specifically cover:

- Followers
- Primitives
- Text Understanding (semantic)
- OCR (verification)
- Line Type Detection
- Layer Assignment
- Symbol Recognition

### 7. Critical Features Deep Dive

#### Text Handling

- Explain the difference between OCR and semantic text understanding
- Provide examples of context-based text determination
- Show how Gemini makes decisions

#### Line Connectivity

- Validation requirements
- How to ensure proper connections
- Gap and overlap handling

#### Symbol Management

- Database-first approach
- Internet search fallback mechanism
- Visual matching strategy

### 8. Data Flow Diagrams

- Show how data moves through the system
- Illustrate decision points
- Map tool execution sequences

### 9. Why the New Approach is Superior to Current System

Comprehensive explanation covering:

- How it improves upon the current generative approach
- Accuracy improvements over existing system
- Fidelity to original drawing (vs. interpretation-based recreation)
- Reduced interpretation errors
- Better line connectivity guarantees
- Intelligent tool selection advantages
- Scalability improvements
- Enhanced quality control

### 10. Implementation Considerations

- Prerequisites
- System requirements
- API integration points
- Database schema considerations
- Performance optimization opportunities
- Migration strategy from current system (if applicable)

### 11. Success Metrics

- Define measurable KPIs
- Quality benchmarks
- Performance targets
- Comparison metrics against current system

### 12. Future Enhancements

- Potential improvements
- Scalability considerations
- Additional features

## Format Requirements

- Use clear markdown formatting
- Include tables where appropriate
- Use code blocks for pseudo-logic examples
- Include ASCII/text diagrams where helpful
- Use hierarchical headings (H1, H2, H3)
- Add bullet points and numbered lists for clarity
- Include visual separators between major sections

## Tone and Style

- Technical but accessible
- Comprehensive and detailed
- Architecture-focused
- Implementation-ready reference document
- No marketing fluff - pure technical documentation

## File Naming

Save the file as: `autocad-rasterization-architecture.md`

## Important Notes

- DO NOT write any code - this is pure documentation
- The current/existing pipeline is ALREADY CODED and working
- Focus on documenting the NEW proposed approach and how it differs
- Be specific about Gemini's decision-making role in the new approach
- Clearly distinguish between the existing generative approach and the new orchestrated approach
- Emphasize that this is a reference document for deciding whether to build the new approach
- Make it comprehensive enough that a developer could understand:
    - What the current system does
    - What the new system would do differently
    - Why the new approach is worth building
    - How to implement it when ready

---

**Remember**: The goal is to create a complete architectural reference document that compares the existing working code against the new proposed orchestrated approach, explaining why and how the new method would be superior, with enough detail to guide the decision of whether to build it and how to implement it when ready.