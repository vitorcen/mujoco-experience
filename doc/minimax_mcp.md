### 可用工具

#### 1. **execute_task**
执行机械臂操控任务。

**参数：**
```json
{
  "task": {
    "type": "string",
    "description": "机械臂操作指令，例如：'open the drawer', 'close the drawer', 'pick up the butter'",
    "required": true
  }
}
```

**示例：**
```json
{
  "task": "put the bowl on the plate"
}
```

#### 2. **get_scene_info**
获取当前场景信息。

**参数：** 无需参数

**返回：**
- 相机捕获的图像
- VLM（视觉语言模型）分析结果
- 物体位置信息
- 抽屉状态
- 机械臂状态

**返回示例：**
```json
{
  "visual_analysis": "场景分析描述...",
  "image_path": "/tmp/scene_capture_xxx.png",
  "verified_task": "验证的任务结果"
}
```

### 注意事项

1. **任务执行规则：**
   - 一次只能执行一个任务
   - 执行后需调用`get_scene_info`验证结果
   - 如果验证失败，可重试一次

2. **不支持的任务：**
   - 关闭任何抽屉 (`close drawer`)
   - 拾取物品 (`pick up`)
   - 未在10个支持任务列表中的任何操作

3. **错误处理：**
   - 当任务执行失败时，系统会返回错误信息
   - 需要配置正确的环境变量（如MINIMAX_API_KEY）

### 当前环境配置问题

从刚才的调用中发现，环境变量`MINIMAX_API_KEY`未设置，这可能导致视觉分析功能受限。

**完整的任务列表（共10个）已在上方列出。** 任何不在此列表中的任务都无法执行。

如需查看具体的任务执行效果，我可以为您演示其中一个任务。您想看哪个任务的执行过程？

✨ Final Result: ## MCP工具接口文档

根据我的系统信息，以下是机械臂控制的MCP接口详情：

### 可用工具

#### 1. **execute_task**
执行机械臂操控任务。

**参数：**
```json
{
  "task": {
    "type": "string",
    "description": "机械臂操作指令，例如：'open the drawer', 'close the drawer', 'pick up the butter'",
    "required": true
  }
}
```

**示例：**
```json
{
  "task": "put the bowl on the plate"
}
```

#### 2. **get_scene_info**
获取当前场景信息。

**参数：** 无需参数

**返回：**
- 相机捕获的图像
- VLM（视觉语言模型）分析结果
- 物体位置信息
- 抽屉状态
- 机械臂状态

**返回示例：**
```json
{
  "visual_analysis": "场景分析描述...",
  "image_path": "/tmp/scene_capture_xxx.png",
  "verified_task": "验证的任务结果"
}
```

### 注意事项

1. **任务执行规则：**
   - 一次只能执行一个任务
   - 执行后需调用`get_scene_info`验证结果
   - 如果验证失败，可重试一次

2. **不支持的任务：**
   - 关闭任何抽屉 (`close drawer`)
   - 拾取物品 (`pick up`)
   - 未在10个支持任务列表中的任何操作

3. **错误处理：**
   - 当任务执行失败时，系统会返回错误信息
   - 需要配置正确的环境变量（如MINIMAX_API_KEY）
