# 微信小程序原型

这是 `Ski_Avatar_Pro` 的微信小程序首版原型，直接复用现有 Python Web API。

## 当前能力
- 选择模式：`JSBA / SAJ / 进阶`
- 直传视频到 `/api/upload`
- 查看任务列表
- 查看已完成结果
- 预览对比视频与压力图

## 如何启动
1. 先启动现有后端：`run_web_app.py`
2. 给后端配置一个固定 HTTPS 域名
3. 把该域名写入 `miniapp_prototype/config.js`
4. 在微信开发者工具中导入 `miniapp_prototype/`
5. 在小程序后台把该域名加入：
   - `request` 合法域名
   - `uploadFile` 合法域名
   - `downloadFile` 合法域名

## 原型说明
- 当前为了快速出样，上传走 `/api/upload` 直传
- 正式上线建议切换到分片上传接口，提升大视频稳定性
- `SAJ` 模式已接到新的 `Infinity` 压力分析风格
