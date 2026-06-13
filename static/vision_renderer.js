// vision_renderer.js - frontend visualisation renderer

class VisionRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.ws = null;
        this.currentData = null;
        
        this.colors = {
            primaryBlue: '#00C8FF',
            secondaryPurple: '#9664FF',
            accentCyan: '#00FFFF',
            white: '#FFFFFF',
            lightGray: '#C8C8C8',
            darkBg: 'rgba(40, 40, 40, 0.8)',
            success: '#7FFF00',
            warning: '#FFA500',
            error: '#FF7272',
            glassBg: 'rgba(20, 20, 20, 0.3)',
        };
        
        this.animations = {
            flashAlpha: 0,
            messageAlpha: 1,
            progressAnimations: {}
        };
        
        this.setupCanvas();
        this.connect();
        this.startRenderLoop();
    }
    
    setupCanvas() {
        const resizeCanvas = () => {
            const rect = this.canvas.getBoundingClientRect();
            this.canvas.width = rect.width;
            this.canvas.height = rect.height;
        };
        
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
    }
    
    connect() {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        this.ws = new WebSocket(`${proto}://${location.host}/ws/vision_data`);
        
        this.ws.onopen = () => {
            console.log('[VisionRenderer] Connected');
            this.updateConnectionStatus(true);
        };
        
        this.ws.onclose = () => {
            console.log('[VisionRenderer] Disconnected');
            this.updateConnectionStatus(false);
            setTimeout(() => this.connect(), 2000);
        };
        
        this.ws.onmessage = (event) => {
            try {
                this.currentData = JSON.parse(event.data);
            } catch (e) {
                console.error('[VisionRenderer] Parse error:', e);
            }
        };
    }
    
    updateConnectionStatus(connected) {
        const badge = document.getElementById('visionStatus');
        if (badge) {
            badge.textContent = connected ? 'Vision: connected' : 'Vision: disconnected';
            badge.className = 'badge ' + (connected ? 'ok' : 'err');
        }
    }
    
    startRenderLoop() {
        const render = () => {
            this.clearCanvas();
            
            if (this.currentData) {
                this.renderFrame(this.currentData);
            }
            
            requestAnimationFrame(render);
        };
        
        render();
    }
    
    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    renderFrame(data) {
        const ctx = this.ctx;
        const W = this.canvas.width;
        const H = this.canvas.height;

        if (data.hand_detected && data.hand_landmarks) {
            this.drawHandSkeleton(data.hand_landmarks);
            if (data.hand_box) {
                this.drawBox(data.hand_box, this.colors.accentCyan, 1);
            }
            this.drawTextWithBg(
                `Grasp Score: ${data.grasp_score.toFixed(2)}`,
                10, 60, 18, this.colors.accentCyan
            );
        }

        if (data.mode === 'SEGMENT' && data.objects) {
            data.objects.forEach((obj, index) => {
                const isSelected = index === data.selected_object_index;
                const color = isSelected ? this.colors.success : this.colors.primaryBlue;
                if (obj.contour) {
                    this.drawContour(obj.contour, color, isSelected ? 3 : 2);
                }
                if (isSelected && obj.center) {
                    this.drawTargetMarker(obj.center.x, obj.center.y);
                }
            });
            if (data.countdown !== null) {
                this.drawCountdown(data.countdown);
            }
        }

        if (data.mode === 'FLASH' && data.flash_progress !== null) {
            this.renderFlashAnimation(data.flash_progress);
        }

        if (data.mode === 'TRACK') {
            if (data.tracking_polygon) {
                this.drawPolygon(data.tracking_polygon, this.colors.success, 2);
            }
            if (data.tracking_center) {
                this.drawCircle(data.tracking_center.x, data.tracking_center.y, 6, this.colors.success);
            }
            if (data.hand_center && data.tracking_center) {
                this.drawMeasureArrow(data.hand_center, data.tracking_center);
            }
            if (data.area_ratio !== null) {
                this.drawAreaRatio(data.area_ratio, data.guidance);
            }
        }

        this.drawTechProgressBars(data.align_score, data.range_score);
        this.drawFPS(data.fps);

        if (data.status_message) {
            this.drawStatusMessage(data.status_message);
        }
    }
    
    drawHandSkeleton(landmarks) {
        const ctx = this.ctx;
        const color = this.colors.secondaryPurple;

        // MediaPipe hand connections
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
            [0, 5], [5, 6], [6, 7], [7, 8],           // index
            [0, 9], [9, 10], [10, 11], [11, 12],      // middle
            [0, 13], [13, 14], [14, 15], [15, 16],    // ring
            [0, 17], [17, 18], [18, 19], [19, 20],    // pinky
            [5, 9], [9, 13], [13, 17]                 // palm
        ];

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        connections.forEach(([i, j]) => {
            if (landmarks[i] && landmarks[j]) {
                ctx.beginPath();
                ctx.moveTo(landmarks[i].x, landmarks[i].y);
                ctx.lineTo(landmarks[j].x, landmarks[j].y);
                ctx.stroke();
            }
        });

        landmarks.forEach(point => {
            this.drawCircle(point.x, point.y, 3, color, true);
        });
    }
    
    drawTextWithBg(text, x, y, fontSize = 18, color = this.colors.white, bgColor = this.colors.glassBg) {
        const ctx = this.ctx;
        const padding = 10;

        ctx.font = `${fontSize}px Arial, sans-serif`;
        const metrics = ctx.measureText(text);
        const textWidth = metrics.width;
        const textHeight = fontSize;

        ctx.fillStyle = bgColor;
        ctx.fillRect(x - padding, y - textHeight - padding,
                     textWidth + 2 * padding, textHeight + 2 * padding);

        ctx.strokeStyle = this.colors.primaryBlue;
        ctx.lineWidth = 1;
        ctx.strokeRect(x - padding, y - textHeight - padding,
                       textWidth + 2 * padding, textHeight + 2 * padding);

        ctx.fillStyle = color;
        ctx.fillText(text, x, y);
    }
    
    drawCountdown(seconds) {
        const text = `Object detected, ${seconds.toFixed(1)}s`;
        const x = 10;
        const y = 100;
        this.drawTextWithBg(text, x, y, 22, this.colors.warning);
    }
    
    renderFlashAnimation(progress) {
        const ctx = this.ctx;
        const W = this.canvas.width;
        const H = this.canvas.height;

        const cycleProgress = progress * 2;
        const alpha = 0.3 + 0.3 * Math.sin(cycleProgress * Math.PI);

        ctx.fillStyle = this.colors.accentCyan + Math.floor(alpha * 255).toString(16).padStart(2, '0');
        ctx.fillRect(0, 0, W, H);

        this.drawTextWithBg('Locking target…',
                           W/2 - 100, H/2, 24, this.colors.accentCyan);
    }
    
    drawTechProgressBars(alignScore, rangeScore) {
        const W = this.canvas.width;
        const H = this.canvas.height;
        const barW = W * 0.3;
        const barH = 8;
        const gap = 20;
        const x0 = 20;
        const y0 = H - 2 * barH - gap - 60;

        this.drawProgressBar(x0, y0, barW, barH, alignScore,
                            'Align', this.colors.primaryBlue);

        this.drawProgressBar(x0, y0 + barH + gap, barW, barH, rangeScore,
                            'Distance(≈1)', this.colors.accentCyan);
    }
    
    drawProgressBar(x, y, width, height, value, label, color) {
        const ctx = this.ctx;

        ctx.fillStyle = this.colors.darkBg;
        ctx.fillRect(x, y, width, height);

        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, width, height);

        const fillWidth = width * Math.max(0, Math.min(1, value));
        if (fillWidth > 0) {
            const gradient = ctx.createLinearGradient(x, y, x + fillWidth, y);
            gradient.addColorStop(0, this.colors.secondaryPurple);
            gradient.addColorStop(1, color);
            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, fillWidth, height);
        }

        this.drawTextWithBg(label, x, y - 10, 14, color);
    }
    
    drawCircle(x, y, radius, color, fill = true) {
        const ctx = this.ctx;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        if (fill) {
            ctx.fillStyle = color;
            ctx.fill();
        } else {
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }
    
    drawBox(box, color, lineWidth = 2) {
        const ctx = this.ctx;
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(box.x, box.y, box.width, box.height);
    }
    
    drawContour(points, color, lineWidth = 2) {
        if (!points || points.length < 3) return;
        
        const ctx = this.ctx;
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.closePath();
        ctx.stroke();
    }
    
    drawPolygon(points, color, lineWidth = 2) {
        this.drawContour(points, color, lineWidth);
    }
    
    drawTargetMarker(x, y) {
        this.drawCircle(x, y, 8, this.colors.success, false);
        this.drawCircle(x, y, 12, this.colors.success, false);
        this.drawTextWithBg('Target', x + 15, y - 5, 16, this.colors.success);
    }
    
    drawMeasureArrow(p1, p2) {
        const ctx = this.ctx;
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        ctx.strokeStyle = this.colors.white;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
        ctx.setLineDash([]);

        const angle = Math.atan2(dy, dx);
        const arrowLength = 15;
        const arrowAngle = Math.PI / 6;
        
        ctx.beginPath();
        ctx.moveTo(p2.x, p2.y);
        ctx.lineTo(
            p2.x - arrowLength * Math.cos(angle - arrowAngle),
            p2.y - arrowLength * Math.sin(angle - arrowAngle)
        );
        ctx.moveTo(p2.x, p2.y);
        ctx.lineTo(
            p2.x - arrowLength * Math.cos(angle + arrowAngle),
            p2.y - arrowLength * Math.sin(angle + arrowAngle)
        );
        ctx.stroke();
        
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        ctx.fillStyle = this.colors.white;
        ctx.font = '14px Arial';
        ctx.fillText(`${distance.toFixed(0)}px`, midX + 10, midY - 10);
    }
    
    drawAreaRatio(ratio, guidance) {
        const y = 120;
        const text = `Area Ratio: ${ratio.toFixed(2)}`;
        this.drawTextWithBg(text, 10, y, 18, this.colors.lightGray);

        if (guidance) {
            const guidanceText = {
                'forward': 'Move Forward',
                'backward': 'Move Back',
                'maintain': 'Maintain'
            };
            const guidanceColor = guidance === 'maintain' ? this.colors.success : this.colors.warning;
            this.drawTextWithBg(guidanceText[guidance] || guidance,
                               10, y + 40, 20, guidanceColor);
        }
    }
    
    drawFPS(fps) {
        const W = this.canvas.width;
        const text = `FPS: ${fps.toFixed(1)}`;
        this.drawTextWithBg(text, W - 120, 30, 16, this.colors.accentCyan);
    }
    
    drawStatusMessage(message) {
        const W = this.canvas.width;
        const H = this.canvas.height;

        let color = this.colors.white;
        if (message.includes('lost') || message.includes('tracking lost')) {
            color = this.colors.error;
        } else if (message.includes('refreshed')) {
            color = this.colors.success;
        }

        this.drawTextWithBg(message, W/2 - 200, H - 50, 20, color);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.visionRenderer = new VisionRenderer('canvas');
}); 