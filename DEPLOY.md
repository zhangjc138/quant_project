# quant_project/DEPLOY.md
# 部署指南

本文档提供quant_project的Docker部署指南。

## 环境要求

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.8+ (本地开发)

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd quant_project
```

### 2. 配置环境

复制配置示例文件：

```bash
cp config.example.json config.json
```

编辑 `config.json`，填入您的API密钥和配置。

### 3. 使用Docker运行

#### 开发模式

```bash
# 构建并启动
docker-compose up --build

# 后台运行
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

#### 生产模式

```bash
# 使用生产配置
docker-compose -f docker-compose.yml up -d --build

# 查看运行状态
docker-compose ps

# 查看实时日志
docker-compose logs -f quant_app
```

### 4. 访问应用

- Web界面: http://localhost:8501
- API文档: http://localhost:8501/api

## 服务说明

### 主应用 (quant_app)

提供Web界面和API服务。

- 端口: 8501
- 容器名: quant_app
- 默认用户: admin
- 默认密码: admin123

### 可选服务

使用 `--profile` 启动可选服务：

```bash
# 启动数据采集服务
docker-compose --profile collector up -d

# 启动回测服务
docker-compose --profile backtest up -d
```

## 数据管理

### 数据目录

- `./data` - 股票数据存储
- `./logs` - 日志文件
- `./output` - 回测结果输出

### 备份数据

```bash
# 备份数据目录
tar -czvf data_backup_$(date +%Y%m%d).tar.gz data/

# 恢复数据
tar -xzvf data_backup_YYYYMMDD.tar.gz
```

## 性能优化

### 生产环境建议

1. **资源限制**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 2G
   ```

2. **健康检查**
   ```yaml
   healthcheck:
     test: ["CMD", "curl", "-f", "http://localhost:8501"]
     interval: 30s
   ```

3. **日志管理**
   - 使用日志驱动: `logging:
       driver: "json-file"`
   - 限制日志大小: `max-size: "10m"`

## 故障排查

### 常见问题

1. **容器启动失败**
   ```bash
   # 查看详细日志
   docker-compose logs quant_app
   
   # 检查配置
   docker-compose config
   ```

2. **端口被占用**
   ```bash
   # 查看端口占用
   lsof -i :8501
   
   # 修改端口
   # 编辑 docker-compose.yml 中的 ports
   ```

3. **数据目录权限问题**
   ```bash
   # 修复权限
   sudo chown -R $USER:$USER data/ logs/ output/
   ```

4. **内存不足**
   ```bash
   # 增加Swap
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### 调试模式

```bash
# 交互式容器
docker-compose run --rm quant_app /bin/bash

# 查看环境变量
docker-compose exec quant_app env
```

## 更新部署

```bash
# 拉取最新代码
git pull

# 重新构建
docker-compose build --no-cache

# 重启服务
docker-compose up -d
```

## 监控

### 健康检查

```bash
# 检查容器健康状态
docker inspect --format='{{.State.Health.Status}}' quant_app
```

### 资源使用

```bash
# 查看资源使用
docker stats
```

## 安全建议

1. **修改默认密码**
   - 首次登录后立即修改密码

2. **使用HTTPS**
   - 配置SSL证书
   - 使用Nginx反向代理

3. **限制访问**
   - 配置防火墙规则
   - 使用VPN或内网访问

## 技术支持

如有问题，请提交Issue或联系维护者。

---

版本: 1.0.0
更新日期: 2024-02-13
