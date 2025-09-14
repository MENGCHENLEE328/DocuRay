# CLAUDE Development Guidelines

## 🀄 Language Policy (Agent Replies)
- Default conversation language: Simplified Chinese (zh-CN).
- All assistant replies to the user should be in Chinese unless the user explicitly requests another language.
- Code comments and internal documentation remain in English per existing rules, unless user explicitly requires Chinese documentation.

## 🚨 CRITICAL TDD WORKFLOW - MUST FOLLOW
**Workflow**: Task Decomposition → TDD Generation → Complete TDD (modify source code, NOT test code) → git commit → git push

### TDD Iron Rules:
1. **🔴 RED**: Write failing tests, define expected behavior
2. **🟢 GREEN**: **Modify production code** to make tests pass (never modify test expectations!)
3. **🔵 REFACTOR**: Optimize production code, keep tests green
4. **❌ FORBIDDEN**: Modifying tests to accommodate code - this is TDD's biggest anti-pattern!

**Test Modification Principle**: Only modify tests when test logic itself is wrong, 99% of the time modify production code!

---

## 🔝 HIGHEST PRIORITY: Git Version Management

### Mandatory Version Numbering System
**EVERY git commit MUST include version number in commit message format:**
```
[vX.Y.Z] commit_message

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Version Numbering Rules
- **Major.Minor.Patch** format (e.g., v1.0.1, v1.0.2, etc.)
- **Major**: Significant architecture changes
- **Minor**: New features or system integrations
- **Patch**: Bug fixes, improvements, incremental development (increment +1)

### Implementation
1. **Before any commit**: Check latest version with `git log --oneline -1`
2. **Increment patch number**: If last commit was v1.0.5, next commit is v1.0.6
3. **Commit format**: Always use `[vX.Y.Z] description` format
4. **No exceptions**: Every commit requires version number

---

## Code Style Requirements

### Core Principles
- **Version**: Each file created/modified should have author, generation event, version number and modification time at top
- **Code Golf**: Minimize lines while maintaining functionality and readability
- **Single Line Comments**: File/function comments must be within one line
- **Unified Configuration**: All variables managed through centralized config files
- **English Only**: All comments and documentation in English (conversation replies are Chinese; see Language Policy)
- **Conservative Modifications**: Only modify code explicitly requested unless comprehensive optimization is specified
- **Documentation Sync**: Update README with any new features/modifications affecting existing operations
- **Systematic Thinking**: Consider entire system and deeply understand essence of each demand
- **First Principle**: Start from function essence rather than existing code
- **DRY Principle**: Any duplicate code found must be pointed out
- **Long-term Consideration**: Assess technical debt and maintenance costs

### Coding Standards

#### Comment Format
```typescript
const result = calculate(input); // Apply calculation logic
function processData(data: Data): Result { } // Process incoming data
```

#### Variable Management
- All configurable values in centralized config files
- No duplicate variable definitions across modules
- Reference config values instead of hardcoding

#### File Organization
- Follow existing project structure patterns
- Place files in appropriate directories according to established patterns

### Development Workflow

#### Test-Driven Development (TDD) Principles - **MANDATORY**
**The Red-Green-Refactor Cycle:**
1. **🔴 RED**: Write a failing test first - test must fail for the right reason
2. **🟢 GREEN**: Write minimal code to make the test pass - no more, no less  
3. **🔵 REFACTOR**: Clean up code while keeping tests green
4. **⚡ VERIFY**: Run all tests after each step to ensure no regressions

**TDD Implementation Rules:**
- **Test First**: NEVER write implementation code without a failing test
- **Minimal Implementation**: Write only enough code to pass the current test
- **Test Verification**: Every test must be run and confirmed to fail before implementation
- **Incremental Development**: Build features test-by-test, not all at once
- **Continuous Testing**: Run tests frequently during development cycle

#### Testing Priority
1. **Test Failures**: First investigate production code issues, not test files
2. **Production Focus**: Tests exist to improve production code quality
3. **Test Modification**: Only modify tests if confirmed to be test-specific issues
4. **Root Cause Analysis**: Always identify underlying production problems

#### Change Management
1. **Original File Priority**: Apply changes to existing files before creating new ones
2. **Fallback Strategy**: Require explicit permission before using fallback approaches
3. **Error Focus**: Summarize tasks by focusing on errors/problems, not successes
4. **Naming Validation**: Check naming conventions for all new file/variable names

### Quality Assurance
- Maintain existing functionality unless explicitly requested to change
- Ensure all modifications follow established patterns
- Validate naming conventions before implementation
- Document changes appropriately in relevant documentation sections

---

## File Naming Conventions

### TypeScript/Python Naming Rules

#### Directory Structure
```
src/
├── services/       # lowercase, plural for service collections
├── models/         # lowercase, plural for model collections
├── utils/          # lowercase, plural for utilities
├── interfaces/     # lowercase, plural for interface collections
└── core/          # lowercase, singular for core modules
```

#### File Naming Patterns
- **Services**: `[domain]_service.py` or `[Domain]Service.ts`
  - ✅ `browser_service.py`, `BrowserService.ts`
  - ❌ `browser-service.py`, `browserService.ts`

- **Interfaces/Types**: `I[Name].ts` or `[domain]_types.py`
  - ✅ `IAgent.ts`, `browser_types.py`
  - ❌ `Agent.interface.ts`, `browser-types.py`

- **Test Files**: `test_[name].py` or `[Name].test.ts`
  - ✅ `test_browser.py`, `Browser.test.ts`
  - ❌ `browser_test.py`, `Browser.spec.ts`

### Variable & Function Naming

#### Variables
- **Python**: snake_case
  - ✅ `browser_session`, `current_state`
  - ❌ `browserSession`, `current-state`
- **TypeScript**: camelCase
  - ✅ `browserSession`, `currentState`
  - ❌ `browser_session`, `current-state`

#### Constants
- **Both**: UPPER_SNAKE_CASE
  - ✅ `MAX_RETRIES`, `DEFAULT_TIMEOUT`
  - ❌ `maxRetries`, `DefaultTimeout`

#### Functions
- **Python**: snake_case (verbs preferred)
  - ✅ `calculate_result()`, `process_data()`
  - ❌ `calculateResult()`, `ProcessData()`
- **TypeScript**: camelCase (verbs preferred)
  - ✅ `calculateResult()`, `processData()`
  - ❌ `calculate_result()`, `ProcessData()`

#### Classes
- **Both**: PascalCase
  - ✅ `BrowserSession`, `EventHandler`
  - ❌ `browserSession`, `event_handler`

### Import/Export Conventions
- **Named exports** for utilities and types
- **Default exports** for classes (when appropriate)
- **Barrel exports** (`__init__.py` or `index.ts`) for public APIs

### File Organization Best Practices
1. **One class/interface per file** (except closely related types)
2. **Group by feature, not by type** in complex modules
3. **Keep test files adjacent** to source files or in dedicated test directories
4. **Use barrel exports** for clean public APIs
5. **Maintain consistent depth** - avoid deeply nested structures

---

## Critical Implementation Rules

### Core Development Principles
1. **Event-Driven Architecture**: Use event systems for decoupled communication
2. **Single Source of Truth**: Maintain clear data ownership boundaries
3. **Error Handling**: Graceful degradation and comprehensive error recovery
4. **Type Safety**: Leverage TypeScript/Python type hints fully
5. **Async First**: Design for asynchronous operations by default

### Data Integrity Principles
1. **Immutable Config**: Configuration from centralized files, not hardcoded
2. **Validation**: Input validation at boundaries
3. **Consistent State**: Ensure state consistency across components
4. **ID Management**: Unique identifier generation and validation

### Testing & Validation
1. **Integration First**: Test component interactions before units
2. **Realistic Mocks**: Test data must reflect actual usage patterns
3. **Edge Cases**: Test boundary conditions and error scenarios
4. **Performance**: Include performance considerations in tests

---

## Documentation Requirements

### Code Documentation
- **Inline Comments**: Brief, single-line explanations for complex logic
- **Function Docstrings**: Clear input/output descriptions
- **Module Headers**: Purpose and key exports at file top
- **TODO Comments**: Track pending improvements with context

### Project Documentation
- **README Updates**: Keep synchronized with implementation
- **API Documentation**: Document all public interfaces
- **Architecture Docs**: Maintain high-level system descriptions
- **Change Logs**: Document significant modifications

---

## Security & Best Practices

### Security Guidelines
1. **No Secrets**: Never commit credentials or API keys
2. **Input Sanitization**: Validate and sanitize all external input
3. **Secure Dependencies**: Regular dependency updates and audits
4. **Least Privilege**: Minimal permissions for all operations

### Performance Considerations
1. **Lazy Loading**: Load resources only when needed
2. **Caching Strategy**: Implement appropriate caching
3. **Resource Cleanup**: Proper disposal of resources
4. **Batch Operations**: Group operations when possible

---

## Development Tools & Commands

### Common Commands
- **Testing**: Run full test suite before commits
- **Linting**: Apply code formatting and style checks
- **Type Checking**: Verify type correctness
- **Pre-commit Hooks**: Automatic quality checks

### Debugging Practices
1. **Structured Logging**: Use appropriate log levels
2. **Error Context**: Include relevant context in errors
3. **Breakpoint Strategy**: Strategic breakpoint placement
4. **Performance Profiling**: Identify bottlenecks early

---

## Collaboration Guidelines

### Code Review Process
1. **Small PRs**: Keep changes focused and reviewable
2. **Clear Descriptions**: Explain what and why
3. **Test Coverage**: Include tests with changes
4. **Documentation**: Update docs with code changes

### Communication
- **Clear Commit Messages**: Descriptive and concise
- **Issue Tracking**: Link commits to issues
- **Progress Updates**: Regular status communication
- **Knowledge Sharing**: Document decisions and learnings
