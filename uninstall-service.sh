#!/bin/bash
# Uninstall script untuk CCTV Vehicle Detection Service

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CCTV Vehicle Detection Service Uninstaller${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    exit 1
fi

SERVICE_NAME="vehicle-detection"

echo -e "${YELLOW}[1/4]${NC} Stopping service..."
systemctl stop ${SERVICE_NAME}.service 2>/dev/null || echo "Service not running"
echo -e "${GREEN}✓${NC} Service stopped"

echo -e "${YELLOW}[2/4]${NC} Disabling service..."
systemctl disable ${SERVICE_NAME}.service 2>/dev/null || echo "Service not enabled"
echo -e "${GREEN}✓${NC} Service disabled"

echo -e "${YELLOW}[3/4]${NC} Removing service file..."
rm -f /etc/systemd/system/${SERVICE_NAME}.service
echo -e "${GREEN}✓${NC} Service file removed"

echo -e "${YELLOW}[4/5]${NC} Reloading systemd..."
systemctl daemon-reload
systemctl reset-failed
echo -e "${GREEN}✓${NC} Systemd reloaded"

echo -e "${YELLOW}[5/5]${NC} Removing service user (optional)..."
read -p "Remove service user 'pengmas-yolo'? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    userdel pengmas-yolo 2>/dev/null || true
    echo -e "${GREEN}✓${NC} User 'pengmas-yolo' removed"
else
    echo -e "${YELLOW}ℹ${NC}  User 'pengmas-yolo' kept (run 'sudo userdel pengmas-yolo' to remove manually)"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Uninstallation completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
