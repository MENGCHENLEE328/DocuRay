/* Entry for DocuRay Agent (TS, Claude Code SDK) */
/* MVP: boot strap + config check; SDK wiring added incrementally */
import fs from 'node:fs';
import path from 'node:path';
function ensureClaudeDir(cwd = process.cwd()) {
    const base = path.join(cwd, '.claude');
    const dirs = [base, path.join(base, 'agents'), path.join(base, 'commands')];
    for (const d of dirs)
        fs.mkdirSync(d, { recursive: true });
    return base;
}
async function main() {
    console.log('DocuRay (macOS) – Claude Code SDK 初始化');
    const base = ensureClaudeDir();
    const settingsPath = path.join(base, 'settings.json');
    if (!fs.existsSync(settingsPath)) {
        const settings = {
            permissionMode: 'strict',
            allowedTools: [
                'mcp:spotlight',
                'mcp:fsevents',
                'mcp:indexer',
                'mcp:search',
                'mcp:preview',
                'mcp:open',
                'shell:read'
            ],
            disallowedTools: ['network:unbounded', 'shell:write', 'git:write']
        };
        fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
        console.log('已创建 .claude/settings.json（严格权限 + 工具白名单）');
    }
    const agentPath = path.join(base, 'agents', 'main.md');
    if (!fs.existsSync(agentPath)) {
        const md = `# DocuRay 主代理（macOS）\n\n- 默认使用中文回复\n- 目标：自然语言→本地文件定位；提供片段级高亮\n- 离线模式：无网时走确定性路由 + 本地检索\n`;
        fs.writeFileSync(agentPath, md, 'utf8');
        console.log('已创建 .claude/agents/main.md');
    }
    const memoryPath = path.join(base, 'CLAUDE.md');
    if (!fs.existsSync(memoryPath)) {
        const memo = `# 项目记忆\n- 对话默认中文\n- macOS 优先：Spotlight/FSEvents/PDFKit/textutil\n- 性能关键路径保持本地实现\n`;
        fs.writeFileSync(memoryPath, memo, 'utf8');
        console.log('已创建 .claude/CLAUDE.md');
    }
    // 提示后续步骤（SDK 连接与 MCP 注册稍后添加）
    console.log('\n下一步：');
    console.log('- 配置 Anthropic API Key（可选）：export ANTHROPIC_API_KEY=...');
    console.log('- 注册本地 MCP：spotlight / indexer / search / preview / open');
    console.log('- 运行 agent：后续将添加 headless 启动脚本与 CLI');
}
main().catch((e) => {
    console.error('启动失败：', e);
    process.exit(1);
});
