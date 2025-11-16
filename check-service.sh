#!/bin/bash
# Check service status and logs

SERVICE_NAME="vehicle-detection"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CCTV Vehicle Detection - Service Check${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

echo -e "${YELLOW}Service Status:${NC}"
systemctl status ${SERVICE_NAME}.service --no-pager
echo ""

echo -e "${YELLOW}Recent Logs (last 50 lines):${NC}"
journalctl -u ${SERVICE_NAME}.service -n 50 --no-pager
echo ""

echo -e "${YELLOW}Service Information:${NC}"
echo "  Enabled: $(systemctl is-enabled ${SERVICE_NAME}.service 2>/dev/null || echo 'no')"
echo "  Active: $(systemctl is-active ${SERVICE_NAME}.service 2>/dev/null || echo 'inactive')"
echo "  Failed: $(systemctl is-failed ${SERVICE_NAME}.service 2>/dev/null || echo 'no')"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "Commands:"
echo -e "  Follow logs: ${YELLOW}sudo journalctl -u ${SERVICE_NAME} -f${NC}"
echo -e "  Restart:     ${YELLOW}sudo systemctl restart ${SERVICE_NAME}${NC}"
echo -e "${GREEN}========================================${NC}"
