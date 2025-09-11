"""
FastAPI web application for GodMode hierarchical reasoning system.

This module creates a modern web application with real-time capabilities
for demonstrating and interacting with hierarchical reasoning models.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

from godmode.core.engine import GodModeEngine
from godmode.experimental.hierarchical_reasoning import HierarchicalReasoningModel
from godmode.web.api import create_api_router
from godmode.web.websocket import WebSocketManager


logger = logging.getLogger(__name__)


class GodModeWebApp:
    """Main web application class for GodMode."""
    
    def __init__(
        self,
        engine: Optional[GodModeEngine] = None,
        hierarchical_model: Optional[HierarchicalReasoningModel] = None,
        debug: bool = False,
    ):
        self.engine = engine or GodModeEngine()
        self.hierarchical_model = hierarchical_model or HierarchicalReasoningModel()
        self.debug = debug
        
        # WebSocket manager for real-time communication
        self.websocket_manager = WebSocketManager()
        
        # Create FastAPI app
        self.app = self._create_app()
        
    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Starting GodMode web application")
            yield
            # Shutdown
            logger.info("Shutting down GodMode web application")
            await self.engine.shutdown()
        
        app = FastAPI(
            title="GodMode: Hierarchical Reasoning System",
            description="Advanced AI reasoning system with hierarchical cognitive architectures",
            version="1.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc",
            lifespan=lifespan,
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files
        # app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Add API routes
        api_router = create_api_router(self.engine, self.hierarchical_model)
        app.include_router(api_router, prefix="/api/v1")
        
        # Add web routes
        self._add_web_routes(app)
        
        # Add WebSocket routes
        self._add_websocket_routes(app)
        
        return app
    
    def _add_web_routes(self, app: FastAPI):
        """Add web page routes."""
        
        templates = Jinja2Templates(directory="templates")
        
        @app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """Home page with overview of the system."""
            return templates.TemplateResponse("home.html", {
                "request": request,
                "title": "GodMode: Hierarchical Reasoning System",
            })
        
        @app.get("/experiment", response_class=HTMLResponse)
        async def experimental_page(request: Request):
            """Experimental page showcasing hierarchical reasoning models."""
            return HTMLResponse(content=self._generate_experimental_page(), status_code=200)
        
        @app.get("/demo", response_class=HTMLResponse)
        async def demo_page(request: Request):
            """Interactive demo page."""
            return templates.TemplateResponse("demo.html", {
                "request": request,
                "title": "GodMode Demo",
            })
        
        @app.get("/visualization", response_class=HTMLResponse)
        async def visualization_page(request: Request):
            """Reasoning visualization page."""
            return templates.TemplateResponse("visualization.html", {
                "request": request,
                "title": "Reasoning Visualization",
            })
        
        @app.get("/analytics", response_class=HTMLResponse)
        async def analytics_page(request: Request):
            """Performance analytics page."""
            return templates.TemplateResponse("analytics.html", {
                "request": request,
                "title": "Performance Analytics",
            })

        @app.get("/system", response_class=HTMLResponse)
        async def system_analysis_page(request: Request):
            """System analysis and optimization page."""
            system_report = self.engine.get_system_report()
            return templates.TemplateResponse("system.html", {
                "request": request,
                "title": "System Analysis",
                "system_report": system_report,
            })

    def _add_websocket_routes(self, app: FastAPI):
        """Add WebSocket routes for real-time communication."""
        
        @app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await self.websocket_manager.connect(websocket, client_id)
            try:
                while True:
                    data = await websocket.receive_json()
                    await self._handle_websocket_message(client_id, data)
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(client_id)
    
    async def _handle_websocket_message(self, client_id: str, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        message_type = data.get("type")
        
        if message_type == "solve_problem":
            # Handle real-time problem solving
            problem_text = data.get("problem", "")
            
            # Send initial response
            await self.websocket_manager.send_personal_message(
                {"type": "reasoning_started", "problem": problem_text},
                client_id
            )
            
            try:
                # Solve problem using hierarchical reasoning
                result = self.hierarchical_model.solve_problem(problem_text)
                
                # Send solution
                await self.websocket_manager.send_personal_message(
                    {
                        "type": "solution_found",
                        "solutions": [s.model_dump() for s in result["solutions"]],
                        "confidence": result["confidence"],
                        "reasoning_trace": result["reasoning_trace"],
                    },
                    client_id
                )
                
            except Exception as e:
                logger.error(f"Error solving problem: {e}")
                await self.websocket_manager.send_personal_message(
                    {"type": "error", "message": str(e)},
                    client_id
                )
        
        elif message_type == "get_statistics":
            # Send engine statistics
            stats = self.engine.get_statistics()
            await self.websocket_manager.send_personal_message(
                {"type": "statistics", "data": stats},
                client_id
            )
    
    def _generate_experimental_page(self) -> str:
        """Generate the experimental page HTML showcasing hierarchical reasoning."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GodMode: Experimental Hierarchical Reasoning</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .experimental-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .card h2 {
            color: #4a5568;
            margin-bottom: 20px;
            font-size: 1.5rem;
        }
        
        .reasoning-levels {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .level {
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .level:hover {
            transform: translateX(10px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .level.metacognitive {
            background: #e6fffa;
            border-color: #38b2ac;
        }
        
        .level.executive {
            background: #f0fff4;
            border-color: #48bb78;
        }
        
        .level.operational {
            background: #fffaf0;
            border-color: #ed8936;
        }
        
        .level.reactive {
            background: #fef5e7;
            border-color: #f6ad55;
        }
        
        .level h3 {
            margin-bottom: 8px;
            font-size: 1.1rem;
        }
        
        .level p {
            font-size: 0.9rem;
            color: #666;
        }
        
        .demo-section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 40px;
            margin-bottom: 30px;
        }
        
        .problem-input {
            width: 100%;
            min-height: 120px;
            padding: 15px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 1rem;
            resize: vertical;
            margin-bottom: 20px;
        }
        
        .solve-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .solve-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }
        
        .solve-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f8fafc;
            border-radius: 10px;
            display: none;
        }
        
        .results.show {
            display: block;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .solution-item {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        
        .confidence-bar {
            width: 100%;
            height: 10px;
            background: #e2e8f0;
            border-radius: 5px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #48bb78, #38b2ac);
            transition: width 1s ease;
        }
        
        .visualization-container {
            grid-column: 1 / -1;
            height: 400px;
            background: white;
            border-radius: 15px;
            padding: 20px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9rem;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Hierarchical Reasoning Models</h1>
            <p>Experimental demonstration of multi-level cognitive architectures</p>
        </div>
        
        <div class="experimental-grid">
            <div class="card">
                <h2>Cognitive Architecture</h2>
                <div class="reasoning-levels">
                    <div class="level metacognitive" onclick="highlightLevel('metacognitive')">
                        <h3>🎯 Metacognitive Level</h3>
                        <p>Strategic planning, problem decomposition, and meta-reasoning about the reasoning process itself.</p>
                    </div>
                    <div class="level executive" onclick="highlightLevel('executive')">
                        <h3>⚡ Executive Level</h3>
                        <p>Goal management, resource allocation, and control of reasoning processes across levels.</p>
                    </div>
                    <div class="level operational" onclick="highlightLevel('operational')">
                        <h3>🔧 Operational Level</h3>
                        <p>Task execution, procedure application, and concrete problem-solving operations.</p>
                    </div>
                    <div class="level reactive" onclick="highlightLevel('reactive')">
                        <h3>⚡ Reactive Level</h3>
                        <p>Immediate responses, pattern matching, and rapid heuristic-based solutions.</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Neural Architecture Features</h2>
                <ul style="list-style: none; padding: 0;">
                    <li style="margin-bottom: 15px;">
                        <strong>🔄 Multi-Level Attention:</strong> Cross-level information flow with hierarchical attention mechanisms
                    </li>
                    <li style="margin-bottom: 15px;">
                        <strong>🧠 Transformer-Based:</strong> Advanced transformer architecture adapted for hierarchical reasoning
                    </li>
                    <li style="margin-bottom: 15px;">
                        <strong>🎛️ Adaptive Gating:</strong> Dynamic control of information flow between cognitive levels
                    </li>
                    <li style="margin-bottom: 15px;">
                        <strong>📊 Confidence Estimation:</strong> Real-time confidence scoring for reasoning quality assessment
                    </li>
                    <li style="margin-bottom: 15px;">
                        <strong>🔗 Cross-Level Communication:</strong> Bidirectional information exchange between reasoning levels
                    </li>
                </ul>
            </div>
        </div>
        
        <div class="demo-section">
            <h2 style="margin-bottom: 20px;">🚀 Interactive Demo</h2>
            <p style="margin-bottom: 20px; color: #666;">Enter a complex problem below to see hierarchical reasoning in action:</p>
            
            <textarea 
                id="problemInput" 
                class="problem-input" 
                placeholder="Example: Plan a sustainable city transportation system that reduces carbon emissions by 50% while maintaining accessibility for all residents..."
            ></textarea>
            
            <button id="solveButton" class="solve-button" onclick="solveProblem()">
                <span id="buttonText">Solve with Hierarchical Reasoning</span>
                <span id="buttonLoading" class="loading" style="display: none;"></span>
            </button>
            
            <div id="results" class="results">
                <h3 style="margin-bottom: 20px;">Reasoning Results</h3>
                <div id="solutionsContainer"></div>
                <div id="reasoningTrace"></div>
            </div>
        </div>
        
        <div class="card visualization-container">
            <h2>Real-time Reasoning Visualization</h2>
            <div id="reasoningViz" style="height: 300px;"></div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" id="totalProblems">0</div>
                <div class="metric-label">Problems Solved</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="avgConfidence">0%</div>
                <div class="metric-label">Average Confidence</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="avgTime">0ms</div>
                <div class="metric-label">Average Solve Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="successRate">0%</div>
                <div class="metric-label">Success Rate</div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let problemsSolved = 0;
        let totalConfidence = 0;
        let totalTime = 0;
        let successfulSolutions = 0;
        
        // Initialize WebSocket connection
        function initWebSocket() {
            const clientId = 'demo_' + Math.random().toString(36).substr(2, 9);
            ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'reasoning_started':
                    showReasoningStarted();
                    break;
                case 'solution_found':
                    showSolutions(data);
                    updateMetrics(data);
                    break;
                case 'error':
                    showError(data.message);
                    break;
            }
        }
        
        function solveProblem() {
            const problemText = document.getElementById('problemInput').value.trim();
            if (!problemText) {
                alert('Please enter a problem to solve');
                return;
            }
            
            const button = document.getElementById('solveButton');
            const buttonText = document.getElementById('buttonText');
            const buttonLoading = document.getElementById('buttonLoading');
            
            button.disabled = true;
            buttonText.style.display = 'none';
            buttonLoading.style.display = 'inline-block';
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'solve_problem',
                    problem: problemText
                }));
            } else {
                // Fallback to HTTP API
                solveProblemHTTP(problemText);
            }
        }
        
        async function solveProblemHTTP(problemText) {
            try {
                const response = await fetch('/api/v1/solve', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        problem: problemText,
                        reasoning_type: 'hierarchical'
                    })
                });
                
                const data = await response.json();
                showSolutions(data);
                updateMetrics(data);
            } catch (error) {
                showError('Failed to solve problem: ' + error.message);
            }
        }
        
        function showReasoningStarted() {
            const results = document.getElementById('results');
            results.innerHTML = '<p>🧠 Hierarchical reasoning in progress...</p>';
            results.classList.add('show');
        }
        
        function showSolutions(data) {
            const results = document.getElementById('results');
            const container = document.getElementById('solutionsContainer');
            
            let html = '';
            if (data.solutions && data.solutions.length > 0) {
                data.solutions.forEach((solution, index) => {
                    html += `
                        <div class="solution-item">
                            <h4>Solution ${index + 1}</h4>
                            <p>${solution.solution_text}</p>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${solution.confidence * 100}%"></div>
                            </div>
                            <small>Confidence: ${(solution.confidence * 100).toFixed(1)}%</small>
                        </div>
                    `;
                });
            } else {
                html = '<p>No solutions generated.</p>';
            }
            
            container.innerHTML = html;
            
            // Show reasoning trace
            if (data.reasoning_trace) {
                const traceHtml = `
                    <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 10px;">
                        <h4>Reasoning Trace</h4>
                        <p><strong>Levels Used:</strong> ${data.reasoning_trace.levels_activated.join(', ')}</p>
                        <p><strong>Processing Steps:</strong> ${data.reasoning_trace.processing_steps}</p>
                        <p><strong>Cross-level Interactions:</strong> ${data.reasoning_trace.cross_level_interactions ? 'Yes' : 'No'}</p>
                    </div>
                `;
                document.getElementById('reasoningTrace').innerHTML = traceHtml;
            }
            
            results.classList.add('show');
            resetSolveButton();
        }
        
        function showError(message) {
            const results = document.getElementById('results');
            results.innerHTML = `<p style="color: red;">Error: ${message}</p>`;
            results.classList.add('show');
            resetSolveButton();
        }
        
        function resetSolveButton() {
            const button = document.getElementById('solveButton');
            const buttonText = document.getElementById('buttonText');
            const buttonLoading = document.getElementById('buttonLoading');
            
            button.disabled = false;
            buttonText.style.display = 'inline';
            buttonLoading.style.display = 'none';
        }
        
        function updateMetrics(data) {
            problemsSolved++;
            if (data.confidence) {
                totalConfidence += data.confidence;
                if (data.confidence > 0.5) successfulSolutions++;
            }
            
            document.getElementById('totalProblems').textContent = problemsSolved;
            document.getElementById('avgConfidence').textContent = 
                Math.round((totalConfidence / problemsSolved) * 100) + '%';
            document.getElementById('successRate').textContent = 
                Math.round((successfulSolutions / problemsSolved) * 100) + '%';
        }
        
        function highlightLevel(levelName) {
            // Remove existing highlights
            document.querySelectorAll('.level').forEach(el => {
                el.style.transform = '';
                el.style.boxShadow = '';
            });
            
            // Highlight selected level
            const level = document.querySelector(`.level.${levelName}`);
            level.style.transform = 'translateX(15px) scale(1.02)';
            level.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.2)';
            
            // Reset after 2 seconds
            setTimeout(() => {
                level.style.transform = '';
                level.style.boxShadow = '';
            }, 2000);
        }
        
        // Initialize visualization
        function initVisualization() {
            const viz = d3.select("#reasoningViz");
            
            // Create a simple network visualization placeholder
            const svg = viz.append("svg")
                .attr("width", "100%")
                .attr("height", "300px");
            
            const width = 800;
            const height = 300;
            
            // Add nodes for each cognitive level
            const levels = [
                {name: "Metacognitive", x: width * 0.2, y: height * 0.2, color: "#38b2ac"},
                {name: "Executive", x: width * 0.8, y: height * 0.2, color: "#48bb78"},
                {name: "Operational", x: width * 0.2, y: height * 0.8, color: "#ed8936"},
                {name: "Reactive", x: width * 0.8, y: height * 0.8, color: "#f6ad55"}
            ];
            
            // Add connections
            svg.selectAll("line")
                .data([
                    {x1: levels[0].x, y1: levels[0].y, x2: levels[1].x, y2: levels[1].y},
                    {x1: levels[0].x, y1: levels[0].y, x2: levels[2].x, y2: levels[2].y},
                    {x1: levels[1].x, y1: levels[1].y, x2: levels[3].x, y2: levels[3].y},
                    {x1: levels[2].x, y1: levels[2].y, x2: levels[3].x, y2: levels[3].y}
                ])
                .enter()
                .append("line")
                .attr("x1", d => d.x1)
                .attr("y1", d => d.y1)
                .attr("x2", d => d.x2)
                .attr("y2", d => d.y2)
                .attr("stroke", "#ccc")
                .attr("stroke-width", 2);
            
            // Add nodes
            svg.selectAll("circle")
                .data(levels)
                .enter()
                .append("circle")
                .attr("cx", d => d.x)
                .attr("cy", d => d.y)
                .attr("r", 30)
                .attr("fill", d => d.color)
                .attr("opacity", 0.8);
            
            // Add labels
            svg.selectAll("text")
                .data(levels)
                .enter()
                .append("text")
                .attr("x", d => d.x)
                .attr("y", d => d.y + 50)
                .attr("text-anchor", "middle")
                .attr("font-size", "12px")
                .attr("fill", "#333")
                .text(d => d.name);
        }
        
        // Initialize everything when page loads
        document.addEventListener('DOMContentLoaded', function() {
            initWebSocket();
            initVisualization();
            
            // Add some example problems
            const examples = [
                "Plan a sustainable city transportation system that reduces carbon emissions by 50% while maintaining accessibility for all residents.",
                "Design an AI system that can learn continuously without forgetting previous knowledge while adapting to new domains.",
                "Develop a strategy to reduce global plastic pollution while maintaining economic growth in developing countries.",
                "Create a distributed computing system that is both highly secure and maximally efficient for real-time applications."
            ];
            
            document.getElementById('problemInput').placeholder = examples[Math.floor(Math.random() * examples.length)];
        });
    </script>
</body>
</html>
        """
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = None,
        **kwargs,
    ):
        """Run the web application."""
        if reload is None:
            reload = self.debug
            
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            **kwargs,
        )


def create_app(
    engine: Optional[GodModeEngine] = None,
    hierarchical_model: Optional[HierarchicalReasoningModel] = None,
    debug: bool = False,
) -> FastAPI:
    """Create a FastAPI application instance."""
    web_app = GodModeWebApp(engine, hierarchical_model, debug)
    return web_app.app
