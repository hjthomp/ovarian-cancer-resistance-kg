
(function () {
    // Check for dependencies (D3.js)
    if (typeof d3 === 'undefined') {
        const msg = "Error: D3.js library not found. Internet connection is required.";
        console.error(msg);
        const loaderText = document.getElementById('loading-text');
        const loaderOverlay = document.getElementById('loading-overlay');
        if (loaderText && loaderOverlay) {
            loaderText.innerHTML = `<strong style="color: #1b1a1a;">${msg}</strong><br><span style="font-size:0.9em">Please check your connection and reload.</span>`;
            loaderOverlay.style.display = 'flex';
        } else {
            alert(msg);
        }
        return;
    }

    // ======= Simulation Constants =======
    const NODE_RADIUS = 8;
    const NODE_RADIUS_MAX_MULTIPLIER = 8;
    const LINK_DISTANCE = 100;
    const STRENGTH = 0.2;
    const CHARGE_STRENGTH = -1000;
    const CLUSTER_STRENGTH = 0.8;
    const MARKER_REF_X = 15; // Original value to show the arrow outside the node
    const ARROW_SIZE = 7; // Normal arrow size
    const HIGHLIGHTED_ARROW_SIZE = 5 // Arrow size when the link is highlighted
    const ZOOM_INITIAL_SCALE = 0.08;
    const ZOOM_MIN = 0, ZOOM_MAX = 3.0;
    const TICK_LABEL_OFFSET_Y = -8;
    const LOD_THRESHOLD = 0.9; // Threshold to switch between Macro and Micro view
    // =====================================

    let svg, simulation, zoom, g, themes, clusterCenters, nodes, links, radiusScale, nodeG, communityHullLayer, pathHighlightLayer, communityLabelLayer, visibleNodeIds; // Graph state variables
    let pathfindingState = { source: null, target: null, selecting: null }; // State for pathfinder
    let articlesMetadata = {}; // Global variable for article metadata
    let topNVisibleNodeIds = null; // For Top N component filter
    let communityLabelFontSize = 56; // Default font size for community labels
    let linkDetailsMap = new Map(); // Optimization: Store heavy evidence data here
    
    // Pagination state
    let filteredLinks = [];
    let currentPage = 1;
    const ITEMS_PER_PAGE = 20;

    // Top Nodes State
    let topNodesState = { active: false, links: [], nodes: [] };
    let activePath = null; // Stores the currently active pathfinding result

    // Global state for layout
    let componentsVisible = false;
    
    // Focus state variables (Moved to top)
    let focusMode = false; 
    let focusRespectsLOD = false;
    let focusBaseScale = 0;
    let expandedNodeIds = new Set(); 
    let expansionSeedNodes = new Set(); 
    let focusedElements = { nodes: new Set(), links: new Set() };

    function showLoader(text) {
        document.getElementById('loading-text').innerText = text;
        document.getElementById('loading-overlay').style.display = 'flex';
        document.body.classList.add('is-loading'); // Block UI interaction
         filteredLinks = [];
         currentPage = 1;
        
        focusMode = false;
        focusRespectsLOD = false;
        expandedNodeIds.clear();
        expansionSeedNodes.clear();
        focusedElements = { nodes: new Set(), links: new Set() };
    }

    function hideLoader() {
        document.getElementById('loading-overlay').style.display = 'none';
        document.body.classList.remove('is-loading'); // Restore UI interaction
    }

    function setActiveTab(button) {
        document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
    }

    // Helper for performance: Debounce function
    function debounce(func, wait) {
        let timeout;
        return function (...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

    // Helper for performance: Throttle function
    function throttle(func, limit) {
        let inThrottle;
        return function (...args) {
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    // --- Helper function for multi-thematic gravity ---
    function getWeightedClusterPos(node, dimension) { // dimension is 'x' or 'y'
        const affinities = node.themeAffinities;
        const width = svg ? +svg.attr("width") : 0;
        if (!affinities || Object.keys(affinities).length === 0) {
            // Fallback for nodes without affinities
            return clusterCenters[node.theme] ? clusterCenters[node.theme][dimension] : width / 2;
        }

        let totalWeight = 0;
        let weightedSum = 0;

        for (const theme in affinities) {
            if (clusterCenters[theme]) { // Ensure the theme has a center
                const weight = affinities[theme];
                totalWeight += weight;
                weightedSum += clusterCenters[theme][dimension] * weight;
            }
        }

        return totalWeight > 0 ? weightedSum / totalWeight : (clusterCenters[node.theme] ? clusterCenters[node.theme][dimension] : width / 2);
    }

    // --- Adapter Layer: Standardize Input Data ---
    function adaptData(rawInput) {
        let graphData = rawInput;
        let metadata = {};

        // Helper to filter metadata to save memory (discard abstract text, etc.)
        const filterMeta = (source) => {
            if (!source) return {};
            const filtered = {};
            for (const key in source) {
                // Only keep fields used in UI
                const item = source[key];
                filtered[key] = {
                    title: item.title,
                    authors: item.authors,
                    year: item.published || item.year,
                    journal: item.journal,
                    doi: item.doi
                };
            }
            return filtered;
        };

        // 1. Detect Format & Extract Metadata
        // Supports:
        // - Unified format: { graph: { theme: [...] }, metadata: { ... } }
        // - Legacy/Simple format: { theme: [...] }
        if (rawInput.graph) {
            graphData = rawInput.graph;
            if (rawInput.metadata) {
                if (rawInput.metadata.processed_articles) {
                    Object.assign(metadata, filterMeta(rawInput.metadata.processed_articles));
                }
                if (rawInput.metadata.processed_abstracts) {
                    Object.assign(metadata, filterMeta(rawInput.metadata.processed_abstracts));
                }
            }
        }

        // 2. Process Graph Data
        const normalizedLinks = [];
        const evidenceMap = new Map();
        const nodeThemeCounts = new Map();
        const processedTriplets = new Set();

        // Helper to count themes for node affinity
        const addThemeCount = (nodeId, theme) => {
            if (!nodeThemeCounts.has(nodeId)) nodeThemeCounts.set(nodeId, {});
            const counts = nodeThemeCounts.get(nodeId);
            counts[theme] = (counts[theme] || 0) + 1;
        };

        // Iterate over themes in the graph data
        const themesList = Object.keys(graphData).sort();
        
        for (const theme of themesList) {
            const triplets = graphData[theme] || [];
            for (const triplet of triplets) {
                const s = triplet.subject;
                const o = triplet.object;
                const rel = triplet.relation ?? triplet.label ?? "";
                
                // Deduplication key
                const tripletKey = `${s}|${rel}|${o}`;
                if (processedTriplets.has(tripletKey)) continue;
                processedTriplets.add(tripletKey);

                // Track theme counts for nodes
                addThemeCount(s, theme);
                addThemeCount(o, theme);

                // Normalize evidence
                const rawEvidence = triplet.evidence_list || triplet.evidence || triplet.references || triplet.Evidence || triplet.Evidence_List || triplet.References;
                let evidence = [];
                if (rawEvidence) {
                    evidence = Array.isArray(rawEvidence) ? rawEvidence : [rawEvidence];
                 }
                
                // Generate unique Link ID
                const linkId = `link_${normalizedLinks.length}`;
                evidenceMap.set(linkId, evidence);

                normalizedLinks.push({
                    id: linkId,
                    source: s, // ID string (D3 will map to objects later)
                    target: o, // ID string
                    relation: rel,
                    source_type: triplet.source_type || 'article',
                    theme: theme
                });
            }
        }

        // 3. Build Nodes
        const normalizedNodes = [];
        nodeThemeCounts.forEach((counts, nodeId) => {
            // Determine primary theme (highest count)
            const primaryTheme = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
            normalizedNodes.push({
                id: nodeId,
                theme: primaryTheme,
                themeAffinities: counts,
                degree: 0 // Initialize degree
            });
        });

        return {
            nodes: normalizedNodes,
            links: normalizedLinks,
            metadata: metadata,
            evidence: evidenceMap,
            themes: themesList
        };
    }

    async function loadGraph(input, button) {
        if (button) setActiveTab(button);

        showLoader('Loading file...');

        // Reset global states for the new graph
        pathfindingState = { source: null, target: null, selecting: null };
        articlesMetadata = {};
        linkDetailsMap.clear();
        // Don't clear data here, only UI state. Data clearing happens in loadGraph.
        filteredLinks = [];
        currentPage = 1;

        document.getElementById('graph').innerHTML = '';
        document.getElementById('minimap').innerHTML = ''; // Clear minimap
        document.getElementById('details').innerHTML = '<i>Click a node or relation to show details here.</i>';
        document.getElementById('themes').innerHTML = '';
        communityLabelFontSize = 56; // Reset font size


        function slugify(s) {
            return String(s || "").replace(/[^a-z0-9]+/ig, "_").replace(/^_|_$/g, "").toLowerCase() || "theme";
        }

        let raw;
        try {
            if (input instanceof File) {
                // Offline Mode: Read local file selected by the user
                const text = await input.text();
                showLoader('Parsing data...');
                raw = JSON.parse(text);
                // Hide landing page if it was visible
                document.getElementById('landing-page').style.display = 'none';
                document.getElementById('graph-container').style.display = 'flex';
            } else {
                // Server Mode: Fetch URL
                const response = await fetch(input);
                if (!response.ok) throw new Error(`File not found or error (${response.status})`);
                showLoader('Parsing data...');
                raw = await response.json();
            }
        } catch (err) {
            document.getElementById("details").innerText = `ERROR loading graph: ${err.message}`;
            console.error(err);
            hideLoader();
            return;
        }

        showLoader('Processing graph...');
        await new Promise(resolve => setTimeout(resolve, 10)); // Allow UI to update

        try {
        // --- OFFLINE MODE TAB SETUP ---
        // If the user loaded a file manually, generate the tab interface here
        if (input instanceof File) {
            const tabsContainer = document.getElementById('tabs');
            tabsContainer.style.display = 'flex';
            tabsContainer.innerHTML = ''; // Clear previous tabs

            // 1. Graph Tab (Active)
            const graphBtn = document.createElement('button');
            graphBtn.className = 'tab-button active';
            graphBtn.textContent = "Graph View";
            graphBtn.onclick = () => {
                setActiveTab(graphBtn);
                document.getElementById('graph-container').style.display = 'flex';
                document.getElementById('metadata-container').style.display = 'none';
                document.getElementById('minimap-container').style.display = 'block';
            };
            tabsContainer.appendChild(graphBtn);

            // 2. Metadata Tab (If it exists in the JSON)
            if (raw.metadata) {
                const metaBtn = document.createElement('button');
                metaBtn.className = 'tab-button';
                metaBtn.textContent = 'Metadata';
                metaBtn.onclick = () => {
                    setActiveTab(metaBtn);
                    document.getElementById('graph-container').style.display = 'none';
                    document.getElementById('metadata-container').style.display = 'block';
                    document.getElementById('minimap-container').style.display = 'none';
                };
                tabsContainer.appendChild(metaBtn);

                // Render metadata
                const metaWrapper = {};
                metaWrapper[input.name] = raw.metadata;
                displayAllMetadata(metaWrapper);
            }

            // Display loaded file name (Visual only)
            const fileLabel = document.createElement('span');
            fileLabel.style.padding = '10px 15px';
            fileLabel.style.fontSize = '13px';
            fileLabel.style.color = '#666';
            fileLabel.style.display = 'flex';
            fileLabel.style.alignItems = 'center';
            fileLabel.innerHTML = `File: <b>${input.name}</b>`;
            tabsContainer.appendChild(fileLabel);

            // 3. Reset Graph Button (Right aligned, first in group)
            const resetGraphBtn = document.createElement('button');
            resetGraphBtn.className = 'tab-button';
            resetGraphBtn.style.marginLeft = 'auto'; // Push this and subsequent buttons to the right
            resetGraphBtn.style.marginRight = '5px';
            resetGraphBtn.innerHTML = '⟳ Reset Graph';
            resetGraphBtn.title = "Reload graph from scratch";
            resetGraphBtn.onclick = () => {
                loadGraph(input);
            };
            tabsContainer.appendChild(resetGraphBtn);

            // 4. Load Different File Button (Next to Reset Graph)
            const loadFileBtn = document.createElement('button');
            loadFileBtn.className = 'tab-button';
            loadFileBtn.textContent = '📂 Load a Different File';
            loadFileBtn.onclick = () => {
                document.getElementById('file-input').click();
            };
            tabsContainer.appendChild(loadFileBtn);
        }
        // ---------------------------------------------------

        // Ensure Graph View is visible before measuring dimensions
        document.getElementById('graph-container').style.display = 'flex';
        document.getElementById('metadata-container').style.display = 'none';
        document.getElementById('minimap-container').style.display = 'block';

        // --- USE ADAPTER ---
        const adapted = adaptData(raw);
        
        // Update Global State
        nodes = adapted.nodes;
        links = adapted.links;
        articlesMetadata = adapted.metadata;
        linkDetailsMap = adapted.evidence;
        themes = adapted.themes;
        visibleNodeIds = new Set();

        // Calculate Degree (Post-processing)
        // Create a map for O(1) lookup instead of O(N) find
        const nodeMap = new Map(nodes.map(n => [n.id, n]));
        links.forEach(link => {
            const sourceNode = nodeMap.get(link.source);
            const targetNode = nodeMap.get(link.target);
            if (sourceNode) sourceNode.degree++;
            if (targetNode) targetNode.degree++;
        });

        // Sort nodes by degree so larger nodes are drawn on top (last in DOM)
        nodes.sort((a, b) => a.degree - b.degree);

        // Scale for node radius based on its degree
        const maxDegree = d3.max(nodes, d => d.degree); // Scale for node radius based on degree
        const minDegree = d3.min(nodes, d => d.degree);
        radiusScale = d3.scaleLinear().domain([minDegree, maxDegree]).range([NODE_RADIUS, NODE_RADIUS * NODE_RADIUS_MAX_MULTIPLIER]); // Adjust range as needed
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10).domain(themes);
        const themeColorMap = {};
        themes.forEach(t => themeColorMap[t] = colorScale(t));
        

        const container = d3.select("#graph");
        let width = container.node().clientWidth;
        let height = container.node().clientHeight;

        svg = container.append("svg")
            .attr("width", width)
            .attr("height", height);

        // Add text indicator to restore view with a clickable link
        const unfocusIndicator = svg.append("text")
            .attr("id", "unfocus-indicator")
            .attr("x", width / 2)
            .attr("y", 30) // Top margin
            .attr("text-anchor", "middle")
            .style("font-size", "14px").style("fill", "rgba(0,0,0,0.5)")
            .style("display", "none"); // Hidden by default

        unfocusIndicator.append("tspan")
            .text("Focus View: Hold ");

        unfocusIndicator.append("tspan")
            .text("Ctrl + Click")
            .style("font-weight", "bold");

        unfocusIndicator.append("tspan")
            .text(" on nodes to expand them. Hold ");

        unfocusIndicator.append("tspan")
            .text("Shift + Drag")
            .style("font-weight", "bold");

        unfocusIndicator.append("tspan")
            .text(" to move the entire focused graph. Press 'Esc' or click ");

        // "here" as a clickable link
        unfocusIndicator.append("tspan")
            .attr("class", "unfocus-link") // Add a class for styling
            .text("here")
            .style("text-decoration", "underline")
            .style("cursor", "pointer")
            .style("fill", "#007bff") // Link color
            .on("click", (event) => {
                event.stopPropagation(); // Prevents the click event from propagating to the SVG
                unfocusAll();
            });

        unfocusIndicator.append("tspan").text(" to restore the full view.");

        g = svg.append("g");
        communityHullLayer = g.append("g").attr("class", "community-hulls"); // Layer for hulls behind everything
        pathHighlightLayer = g.append("g").attr("class", "path-highlights"); // Layer for path highlights (behind links)

        const defs = svg.append("defs");
        themes.forEach(th => {
            const id = "arrow_" + slugify(th);
            const c = themeColorMap[th];
            defs.append("marker")
                .attr("id", id)
                .attr("viewBox", "0 -5 10 10") // Define the visible region of the marker
                .attr("refX", MARKER_REF_X)
                .attr("refY", 0) // Center the arrow vertically
                .attr("markerWidth", ARROW_SIZE)
                .attr("markerHeight", ARROW_SIZE)
                .attr("orient", "auto")
                .append("path")
                .attr("d", "M0,-5L10,0L0,5")
                .attr("fill", c);
        });
        // Define arrow markers for highlighted links
        themes.forEach(th => {
            const highlightedId = "arrow_highlighted_" + slugify(th);
            const c = themeColorMap[th];
            defs.append("marker")
                .attr("id", highlightedId)
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", MARKER_REF_X) // Use the same refX to maintain relative position
                .attr("refY", 0)
                .attr("markerWidth", HIGHLIGHTED_ARROW_SIZE)
                .attr("markerHeight", HIGHLIGHTED_ARROW_SIZE)
                .attr("orient", "auto")
                .append("path")
                .attr("d", "M0,-5L10,0L0,5")
                .attr("fill", c);
        });

        clusterCenters = {};
        const margin = 80;
        themes.forEach((th, i) => {
            const x = margin + i * (Math.max(width - 2 * margin, 0)) / Math.max(themes.length - 1, 1);
            clusterCenters[th] = { x: x, y: height / 2 };
        });

        simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(LINK_DISTANCE).strength(STRENGTH))
            .force("charge", d3.forceManyBody().strength(CHARGE_STRENGTH))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide(d => radiusScale(d.degree) + 5))
            .force("clusterX", d3.forceX(d => getWeightedClusterPos(d, 'x')).strength(CLUSTER_STRENGTH))
            .force("clusterY", d3.forceY(d => getWeightedClusterPos(d, 'y')).strength(CLUSTER_STRENGTH))
            .velocityDecay(0.4) // Menor fricción para permitir mejor movimiento inicial
            .alphaDecay(0.1);   // Enfriamiento más lento para mejor organización

        showLoader('Loading...');        
        await new Promise(resolve => setTimeout(resolve, 10)); // Allow UI to update

        // Pre-calculate layout to avoid initial animation lag
        // simulation.stop();
        // for (let i = 0; i < 100; ++i) simulation.tick();
        simulation.restart();

        // --- Create links ---
        const link = g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("class", "link")
            .attr("stroke", d => themeColorMap[d.theme] || "#666")
            .attr("stroke-dasharray", d => d.source_type === 'article' ? null : "5,5") // Dashed line for augmented links
            .attr("stroke-width", 2)
            .attr("marker-end", d => `url(#arrow_${slugify(d.theme)})`) // Default normal arrow
            .on("click", function (event, d) {
                // Reset the state of all links
                g.selectAll(".link")
                    .classed("highlighted", false)
                    .attr("marker-end", l => `url(#arrow_${slugify(l.theme)})`); // Revert to normal arrow

                // Highlight the clicked link
                d3.select(this)
                    .classed("highlighted", true)
                    .attr("marker-end", `url(#arrow_highlighted_${slugify(d.theme)})`); // Use highlighted arrow
                focusOnLink(d);
                showLinkDetails(d); 
                event.stopPropagation();
            })
            .on("mouseenter", function () { d3.select(this).attr("opacity", 1); })
            .on("mouseleave", function () { d3.select(this).attr("opacity", null); });

        // --- Create link labels ---
        const linkLabels = g.append("g")
            .attr("class", "link-labels")
            .selectAll("text")
            .data(links)
            .join("text")
            .attr("class", "link-label")
            .style("cursor", "pointer") // Change cursor to indicate it's clickable
            .style("user-select", "none")
            .attr("text-anchor", "middle")
            .on("click", function (event, d) {
                // Reset the state of all links
                g.selectAll(".link")
                    .classed("highlighted", false)
                    .attr("marker-end", l => `url(#arrow_${slugify(l.theme)})`); // Revert to normal arrow

                // Highlight the corresponding link
                link.filter(ld => ld === d) // Filter to find the link line
                    .classed("highlighted", true)
                    .attr("marker-end", `url(#arrow_highlighted_${slugify(d.theme)})`); // Use highlighted arrow

                focusOnLink(d);
                showLinkDetails(d);
                event.stopPropagation();
            })
            .on("mouseenter", function () { d3.select(this).attr("opacity", 1); })
            .on("mouseleave", function () { d3.select(this).attr("opacity", null); })
            .text(d => d.relation);

        // --- Create nodes ---
        nodeG = g.append("g")
            .selectAll(".node")
            .data(nodes)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended)
            )
            .on("click", function (event, d) {
                if (pathfindingState.selecting) {
                    setPathNode(d, pathfindingState.selecting);
                    event.stopPropagation();
                    return;
                }
                if (event.ctrlKey) {
                    expandNode(d); // Expand view with Ctrl+Click
                } else {
                    focusOnNode(d); // Normal focus behavior
                }
                showNodeDetails(d);
                event.stopPropagation();
            })
            .on("dblclick", (event, d) => {
                // On double click, release the node so the simulation can move it
                d.fx = null;
                d.fy = null;
            });

        nodeG.append("circle")
            .attr("r", d => radiusScale(d.degree)) // Use the scaled radius
            .attr("fill", d => themeColorMap[d.theme] || "#999");

        nodeG.append("text")
            .attr("dy", d => radiusScale(d.degree) + 2) // Adjust text position based on the new radius
            .attr("text-anchor", "middle")
            .style("font-size", d => Math.max(10, radiusScale(d.degree) * 0.6) + "px")
            .style("user-select", "none")
            .text(d => d.id);

        nodeG.append("title")
            .text(d => `${d.id}\nConnections: ${d.degree}\nTheme: ${d.theme}`);

        communityLabelLayer = g.append("g").attr("class", "community-labels"); // Layer for labels on top
        // --- Zoom Behavior (Initialized AFTER elements are created) ---
        zoom = d3.zoom().scaleExtent([ZOOM_MIN, ZOOM_MAX]).on("zoom", (event) => {
            g.attr("transform", event.transform);
            
            // Update Level of Detail (LOD) based on zoom level
            requestAnimationFrame(() => updateLOD(event.transform));

            // Update minimap on each zoom/pan event
            if (window.updateMinimapViewbox) {
                window.updateMinimapViewbox(event.transform);
            }
        });

        svg.call(zoom);

        // Inicializar zoom AHORA que todos los elementos (links, nodes) existen
        svg.call(zoom.scaleTo, ZOOM_INITIAL_SCALE);
        
        // Define tick function to be used by simulation and manual updates
        function ticked() {
            // Optimization: Use simple center-to-center links while moving fast
            const isHighPerformance = simulation.alpha() > 0.05;

            // Calculate adjusted start and end points for links
            // so they don't overlap with nodes and arrows are displayed correctly.
            link
                .attr("x1", d => {
                    if (isHighPerformance) return d.source.x;
                    const sourceRadius = radiusScale(d.source.degree);
                    const dx = d.target.x - d.source.x;
                    const dy = d.target.y - d.source.y;
                    const L = Math.hypot(dx, dy);
                    if (L === 0) return d.source.x; // Avoid division by zero if nodes coincide
                    // Adjust x1 to start at the source node's circumference
                    return d.source.x + (dx / L) * sourceRadius;
                })
                .attr("y1", d => {
                    if (isHighPerformance) return d.source.y;
                    const sourceRadius = radiusScale(d.source.degree);
                    const dx = d.target.x - d.source.x;
                    const dy = d.target.y - d.source.y;
                    const L = Math.hypot(dx, dy);
                    if (L === 0) return d.source.y; // Avoid division by zero
                    // Adjust y1 to start at the source node's circumference
                    return d.source.y + (dy / L) * sourceRadius;
                })
                .attr("x2", d => {
                    if (isHighPerformance) return d.target.x;
                    const targetRadius = radiusScale(d.target.degree);
                    const dx = d.target.x - d.source.x;
                    const dy = d.target.y - d.source.y;
                    const L = Math.hypot(dx, dy);
                    if (L === 0) return d.target.x; // Avoid division by zero
                    // Adjust x2 to end at the target node's circumference
                    return d.target.x - (dx / L) * targetRadius;
                })
                .attr("y2", d => {
                    if (isHighPerformance) return d.target.y;
                    const targetRadius = radiusScale(d.target.degree);
                    const dx = d.target.x - d.source.x;
                    const dy = d.target.y - d.source.y;
                    const L = Math.hypot(dx, dy);
                    if (L === 0) return d.target.y; // Avoid division by zero
                    // Adjust y2 to end at the target node's circumference
                    return d.target.y - (dy / L) * targetRadius;
                });

            linkLabels
                .attr("x", (d) => (d.source.x + d.target.x) / 2)
                .attr("y", (d) => (d.source.y + d.target.y) / 2 + TICK_LABEL_OFFSET_Y);

            nodeG.attr("transform", d => `translate(${d.x},${d.y})`);

            // Update path highlights
            pathHighlightLayer.selectAll("path").attr("d", d => {
                const source = d.link.source;
                const target = d.link.target;

                const offset = d.offset || 0;

                if (offset === 0) {
                    return `M${source.x},${source.y}L${target.x},${target.y}`;
                }

                const dx = target.x - source.x;
                const dy = target.y - source.y;
                const len = Math.sqrt(dx * dx + dy * dy) || 1;

                const nx = -dy / len;
                const ny = dx / len;

                const offX = nx * offset;
                const offY = ny * offset;

                return `M${source.x + offX},${source.y + offY}L${target.x + offX},${target.y + offY}`;
            });

            // Update loader progress
            if (simulation.alpha() > 0.01) {
                const progress = Math.min(100, Math.floor((1 - simulation.alpha()) * 100));
                const loaderText = document.getElementById('loading-text');
                if (loaderText && document.getElementById('loading-overlay').style.display !== 'none') {
                    loaderText.innerText = `Loading... ${progress}%`;
                }
            }
        
            // Update community visuals if active
            if (componentsVisible && window.updateCommunityVisuals) {
                window.updateCommunityVisuals();
            }
        }

        simulation.on("tick", ticked);
        simulation.on("end", () => hideLoader());

        // Expose ticked for global resize
        window.ticked = ticked;

        // --- Drag Functions ---
        function dragstarted(event, d) {
            // If in focus mode and Shift is pressed, prepare for group drag
            if (focusMode && event.sourceEvent.shiftKey) {
                const visibleNodes = getVisibleNodes();
                visibleNodes.forEach(node => {
                    // Save the initial position of each node in the group
                    node.startX = node.x;
                    node.startY = node.y;
                });
                d.groupDragStartX = event.x; // Save the initial X coordinate of the mouse for the group
                d.groupDragStartY = event.y; // Save the initial Y coordinate of the mouse for the group
            }
            // Do NOT restart simulation on drag start
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            // If in focus mode, move the entire group
            if (focusMode && event.sourceEvent.shiftKey) { // Move the entire group only if Shift is pressed
                const dx = event.x - d.groupDragStartX; // Calculate delta from the start of the group drag
                const dy = event.y - d.groupDragStartY;
                const visibleNodes = getVisibleNodes();
                visibleNodes.forEach(node => {
                    node.fx = node.startX + dx; // Apply delta to the initial position of each node
                    node.fy = node.startY + dy;
                    node.x = node.fx; // Manually sync x/y for immediate render
                    node.y = node.fy;
                });
            } else { // Normal drag behavior
                d.fx = event.x;
                d.fy = event.y;
                d.x = event.x; // Manually sync x/y for immediate render
                d.y = event.y;
            }
            ticked(); // Manually update DOM without running simulation
        }

        function dragended(event, d) {
            // Do NOT change alphaTarget on drag end
            // Clear the group drag start coordinates
            delete d.groupDragStartX;
            delete d.groupDragStartY;
            // We don't release the node here, so it stays fixed.
        }

        // --- Community Drag Functions ---
        function communityDragStarted(event, d) {
            d.members.forEach(node => {
                node.startX = node.x;
                node.startY = node.y;
                node.fx = node.x;
                node.fy = node.y;
            });
            d.dragStartX = event.x;
            d.dragStartY = event.y;
        }

        function communityDragged(event, d) {
            const dx = event.x - d.dragStartX;
            const dy = event.y - d.dragStartY;
            d.members.forEach(node => {
                node.fx = node.startX + dx;
                node.fy = node.startY + dy;
                node.x = node.fx;
                node.y = node.fy;
            });
            ticked();
        }

        // --- Focus and Context Functions ---
        const FADE_OPACITY = 0.1; // Opacity for faded elements

        function focusOnNode(d) {
            // Show the message to restore the view
            svg.select("#unfocus-indicator").style("display", "block");

            const connectedNodeIds = new Set();
            focusMode = true; // Activate focus mode
            focusRespectsLOD = false;
            expandedNodeIds.clear(); // Clear previous expansion
            expansionSeedNodes.clear();
            focusedElements.nodes.clear();
            focusedElements.links.clear();

            focusedElements.nodes.add(d.id);

            // Find all directly connected nodes
            links.forEach(function (l) {
                if (l.source.id === d.id) {
                    focusedElements.nodes.add(l.target.id);
                    focusedElements.links.add(l);
                }
                if (l.target.id === d.id) {
                    focusedElements.nodes.add(l.source.id);
                    focusedElements.links.add(l);
                }
            });

            // Fade out nodes that are not connected
            nodeG.style("opacity", n => focusedElements.nodes.has(n.id) ? 1 : FADE_OPACITY)
                .style("pointer-events", n => focusedElements.nodes.has(n.id) ? "auto" : "none");

            // Hide labels of faded nodes and disable their events to override CSS
            nodeG.select("text").style("opacity", n => focusedElements.nodes.has(n.id) ? 1 : 0)
                .style("pointer-events", n => focusedElements.nodes.has(n.id) ? "auto" : "none");

            // Fade out links that are not connected to the main node
            updateLinkAndLabelVisibility(l => focusedElements.links.has(l));

            // Update minimap to reflect focus state
            if (window.updateMinimapNodes) window.updateMinimapNodes();
            updateLOD(); // Ensure visibility is updated
        }

        function expandNode(d) {
            if (!focusMode) {
                svg.select("#unfocus-indicator").style("display", "block");
                focusMode = true;
                // Clear simple focus if it existed
                focusedElements.nodes.clear();
                focusedElements.links.clear();
            }
            focusRespectsLOD = false;

            // Toggle node 'd' in the expansion seed set
            if (expansionSeedNodes.has(d.id)) {
                expansionSeedNodes.delete(d.id); // If it's already there, remove it (collapse)
            } else {
                expansionSeedNodes.add(d.id); // If not, add it (expand)
            }

            // Clear and recalculate all visible nodes from scratch
            expandedNodeIds.clear();

            if (expansionSeedNodes.size === 0) {
                // If no seeds are left, return to the initial focus view
                unfocusAll();
                return;
            }

            // Add all seeds and their direct neighbors to the set of visible nodes
            expansionSeedNodes.forEach(seedId => {
                expandedNodeIds.add(seedId);
                links.forEach(l => {
                    if (l.source.id === seedId) expandedNodeIds.add(l.target.id);
                    if (l.target.id === seedId) expandedNodeIds.add(l.source.id);
                });
            });

            // Update visibility based on the recalculated set
            nodeG.style("opacity", n => expandedNodeIds.has(n.id) ? 1 : FADE_OPACITY)
                .style("pointer-events", n => expandedNodeIds.has(n.id) ? "auto" : "none");
            nodeG.select("text").style("opacity", n => expandedNodeIds.has(n.id) ? 1 : 0)
                .style("pointer-events", n => expandedNodeIds.has(n.id) ? "auto" : "none");
            updateLinkAndLabelVisibility(l => expandedNodeIds.has(l.source.id) && expandedNodeIds.has(l.target.id));

            // Update minimap to reflect focus state
            if (window.updateMinimapNodes) window.updateMinimapNodes();
            updateLOD();
        }

        function focusOnLink(d, force = false) {
            // If we are already in focus mode, we don't alter node visibility.
            // We just let the link's click handler highlight it.
            if (focusMode && !force) {
                return; // Do nothing else to avoid losing the current expansion.
            }

            // If we weren't in focus mode, proceed to focus only on this link.
            svg.select("#unfocus-indicator").style("display", "block");
            focusMode = true; // Activate focus mode
            focusRespectsLOD = false;
            expandedNodeIds.clear();
            expansionSeedNodes.clear();

            focusedElements.nodes.clear();
            focusedElements.links.clear();
            focusedElements.nodes.add(d.source.id);
            focusedElements.nodes.add(d.target.id);
            focusedElements.links.add(d);

            nodeG.style("opacity", n => focusedElements.nodes.has(n.id) ? 1 : FADE_OPACITY)
                .style("pointer-events", n => focusedElements.nodes.has(n.id) ? "auto" : "none");
            nodeG.select("text").style("opacity", n => focusedElements.nodes.has(n.id) ? 1 : 0)
                .style("pointer-events", n => focusedElements.nodes.has(n.id) ? "auto" : "none");

            updateLinkAndLabelVisibility(l => l === d);

            // Update minimap to reflect focus state
            if (window.updateMinimapNodes) window.updateMinimapNodes();
            updateLOD();
        }

        function focusOnCommunity(d) {
            // Instead of isolating (Focus Mode), just zoom to the community.
            // This respects the LOD logic: links will appear naturally if we get close enough.

            // Calculate bounding box of the community
            let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
            d.members.forEach(n => {
                if (n.x < x0) x0 = n.x;
                if (n.x > x1) x1 = n.x;
                if (n.y < y0) y0 = n.y;
                if (n.y > y1) y1 = n.y;
            });

            const width = +svg.attr("width");
            const height = +svg.attr("height");
            const dx = x1 - x0;
            const dy = y1 - y0;
            const cx = (x0 + x1) / 2;
            const cy = (y0 + y1) / 2;

            // Calculate scale to fit with padding (0.8 factor)
            let scale = 0.8 / Math.max(dx / width, dy / height);
            scale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, scale));

            focusBaseScale = scale;

            const transform = d3.zoomIdentity
                .translate(width / 2, height / 2)
                .scale(scale)
                .translate(-cx, -cy);

            svg.transition().duration(750).call(zoom.transform, transform);

            // If nodes are hidden, reveal this community in Focus Mode
            if (d3.select("#hide-nodes").property("checked")) {
                svg.select("#unfocus-indicator").style("display", "block");
                focusMode = true;
                focusRespectsLOD = true;

                expandedNodeIds.clear();
                expansionSeedNodes.clear();
                focusedElements.nodes.clear();
                focusedElements.links.clear();

                const memberIds = new Set(d.members.map(n => n.id));
                d.members.forEach(n => focusedElements.nodes.add(n.id));

                // Add internal links
                links.forEach(l => {
                    if (memberIds.has(l.source.id) && memberIds.has(l.target.id)) {
                        focusedElements.links.add(l);
                    }
                });

                // Update visuals (Keep background nodes hidden with opacity 0)
                nodeG.style("opacity", n => focusedElements.nodes.has(n.id) ? 1 : 0)
                    .style("pointer-events", n => focusedElements.nodes.has(n.id) ? "auto" : "none");

                nodeG.select("text").style("opacity", n => focusedElements.nodes.has(n.id) ? 1 : 0)
                    .style("pointer-events", n => focusedElements.nodes.has(n.id) ? "auto" : "none");

                updateLinkAndLabelVisibility(l => focusedElements.links.has(l));

                if (window.updateMinimapNodes) window.updateMinimapNodes();
                updateLOD();
            }
        }

        function drawPathHighlights(highlightGroups) {
            // Detect shared links and create offsets
            const linkColors = new Map(); // link -> Array of colors

            highlightGroups.forEach(group => {
                group.links.forEach(l => {
                    if (!linkColors.has(l)) linkColors.set(l, []);
                    linkColors.get(l).push(group.color);
                });
            });

            const finalHighlights = [];
            const offsetStep = 5; // Spacing between parallel lines

            linkColors.forEach((colors, l) => {
                const count = colors.length;
                colors.forEach((color, i) => {
                    // Calculate offset to center the group of lines
                    const offset = (i - (count - 1) / 2) * offsetStep;
                    finalHighlights.push({ link: l, color: color, offset: offset });
                });
            });

            pathHighlightLayer.selectAll("path")
                .data(finalHighlights)
                .join("path")
                .attr("class", "path-highlight")
                .attr("stroke", d => d.color)
                .attr("stroke-width", 4)
                .attr("opacity", 0.7);

            // Force a tick update to render the new paths immediately
            ticked();
        }

        function focusOnPath(pathNodes, pathLinks) {
            svg.select("#unfocus-indicator").style("display", "block");
            focusMode = true;
            focusRespectsLOD = false;
            expandedNodeIds.clear();
            expansionSeedNodes.clear();
            focusedElements.nodes.clear();
            focusedElements.links.clear();

            pathNodes.forEach(n => focusedElements.nodes.add(n.id));
            pathLinks.forEach(l => focusedElements.links.add(l));

            nodeG.style("opacity", n => focusedElements.nodes.has(n.id) ? 1 : FADE_OPACITY)
                .style("pointer-events", n => focusedElements.nodes.has(n.id) ? "auto" : "none");
            nodeG.select("text").style("opacity", n => focusedElements.nodes.has(n.id) ? 1 : 0)
                .style("pointer-events", n => focusedElements.nodes.has(n.id) ? "auto" : "none");

            updateLinkAndLabelVisibility(l => focusedElements.links.has(l));

            // Update minimap
            if (window.updateMinimapNodes) window.updateMinimapNodes();
            updateLOD();

            // Auto-zoom to fit the path
            if (pathNodes.length > 0) {
                const width = +svg.attr("width");
                const height = +svg.attr("height");
                let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;

                pathNodes.forEach(n => {
                    if (n.x < x0) x0 = n.x;
                    if (n.x > x1) x1 = n.x;
                    if (n.y < y0) y0 = n.y;
                    if (n.y > y1) y1 = n.y;
                });

                const dx = x1 - x0;
                const dy = y1 - y0;
                const cx = (x0 + x1) / 2;
                const cy = (y0 + y1) / 2;

                // Calculate scale with padding (0.6 factor)
                let scale = 0.6 / Math.max(dx / width, dy / height);
                scale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, scale));

                const transform = d3.zoomIdentity
                    .translate(width / 2, height / 2)
                    .scale(scale)
                    .translate(-cx, -cy);

                svg.transition().duration(750).call(zoom.transform, transform);
            }
        }

        function focusTopNodes(n) {
            // Sort nodes by degree
            const sorted = [...nodes].sort((a, b) => b.degree - a.degree);
            const top = sorted.slice(0, n);
            
            if (top.length === 0) return;

            svg.select("#unfocus-indicator").style("display", "block");
            focusMode = true;
            focusRespectsLOD = false;
            
            // Clear previous states
            expandedNodeIds.clear();
            expansionSeedNodes.clear();
            focusedElements.nodes.clear();
            focusedElements.links.clear();
            
            // Add to seeds
            top.forEach(node => expansionSeedNodes.add(node.id));
            
            // Calculate expansion (Seeds + Neighbors)
            expansionSeedNodes.forEach(seedId => {
                expandedNodeIds.add(seedId);
                links.forEach(l => {
                    if (l.source.id === seedId) expandedNodeIds.add(l.target.id);
                    if (l.target.id === seedId) expandedNodeIds.add(l.source.id);
                });
            });

            // Update visuals
            nodeG.style("opacity", n => expandedNodeIds.has(n.id) ? 1 : FADE_OPACITY)
                .style("pointer-events", n => expandedNodeIds.has(n.id) ? "auto" : "none");
            
            nodeG.select("text").style("opacity", n => expandedNodeIds.has(n.id) ? 1 : 0)
                .style("pointer-events", n => expandedNodeIds.has(n.id) ? "auto" : "none");
                
            // Create a more specific set of links to show, focusing on connections from the hubs
            links.forEach(l => {
                // Show link if it's connected to a top node (a seed)
                if (expansionSeedNodes.has(l.source.id) || expansionSeedNodes.has(l.target.id)) {
                    focusedElements.links.add(l);
                }
            });

            updateLinkAndLabelVisibility(l => focusedElements.links.has(l));
            if (window.updateMinimapNodes) window.updateMinimapNodes();
            updateLOD();
            
            // Setup Top Nodes List State for Side Panel
            const relevantLinks = Array.from(focusedElements.links);
            
            // Sort: Internal links (between top nodes) first
            const nodeIds = expansionSeedNodes;
            relevantLinks.sort((a, b) => {
                const aInternal = nodeIds.has(a.source.id) && nodeIds.has(a.target.id);
                const bInternal = nodeIds.has(b.source.id) && nodeIds.has(b.target.id);
                if (aInternal && !bInternal) return -1;
                if (!aInternal && bInternal) return 1;
                return 0;
            });

            topNodesState.active = true;
            topNodesState.links = relevantLinks;
            topNodesState.nodes = top;
            currentPage = 1;
            renderGlobalList();
        }

        function getVisibleNodes() {
            const visibleIds = expansionSeedNodes.size > 0 ? expandedNodeIds : focusedElements.nodes;
            return nodes.filter(n => visibleIds.has(n.id));
        }

        function updateLinkAndLabelVisibility(visibilityCheck) {
            link.style("opacity", l => visibilityCheck(l) ? 1 : FADE_OPACITY)
                .style("pointer-events", l => visibilityCheck(l) ? "auto" : "none");

            linkLabels.style("opacity", l => visibilityCheck(l) ? 1 : FADE_OPACITY)
                .style("pointer-events", l => visibilityCheck(l) ? "auto" : "none");
        }

        function unfocusAll() {
            // Hide the message to restore the view
            svg.select("#unfocus-indicator").style("display", "none");
            topNodesState.active = false;
            d3.select("#show-top-nodes").property("checked", false);
            
            // Remove specific link highlights (e.g. from clicking a card)
            g.selectAll(".link")
                .classed("highlighted", false)
                .attr("marker-end", l => `url(#arrow_${slugify(l.theme)})`); // Revert to normal arrow

            // Check if we should restore path view instead of full reset
            // We restore if an active path exists AND we are currently looking at a subset (e.g. single link details)
            if (activePath && (focusedElements.nodes.size !== activePath.nodes.length || focusedElements.links.size !== activePath.links.length)) {
                focusOnPath(activePath.nodes, activePath.links);
                return;
            }

            // Clear highlights
            pathHighlightLayer.selectAll("*").remove();

            focusMode = false; // Deactivate focus mode
            focusRespectsLOD = false;
            activePath = null; // Clear active path

            expandedNodeIds.clear(); // Clear the set of expanded nodes
            expansionSeedNodes.clear(); // Also clear the seeds
            focusedElements.nodes.clear();
            focusedElements.links.clear();

            // If returning to component view, re-run the layout to apply all filters (like Top N)
            if (componentsVisible) {
                updateComponentLayout(true);
            } else {
                // If returning to theme view, reset opacity and update filters
                nodeG.style("opacity", 1).style("pointer-events", "auto");
                nodeG.select("text").style("opacity", 1).style("pointer-events", "auto");
                link.style("opacity", null).style("pointer-events", "auto");
                linkLabels.style("opacity", 1).style("pointer-events", "auto");
                updateFilters();
            }

            // Release all nodes (unpin) so they react to simulation forces again
            nodes.forEach(d => {
                d.fx = null;
                d.fy = null;
            });
            // simulation.alpha(0.1).restart(); // Removed to prevent movement when exiting focus

            // Update minimap to reflect focus state
            if (window.updateMinimapNodes) window.updateMinimapNodes();
            updateLOD();
        }

        function zoomToLink(d) {
            const width = +svg.attr("width");
            const height = +svg.attr("height");
            
            // Calculate bounding box of the two nodes
            const x0 = Math.min(d.source.x, d.target.x);
            const x1 = Math.max(d.source.x, d.target.x);
            const y0 = Math.min(d.source.y, d.target.y);
            const y1 = Math.max(d.source.y, d.target.y);

            const dx = x1 - x0;
            const dy = y1 - y0;
            const cx = (x0 + x1) / 2;
            const cy = (y0 + y1) / 2;
            
            // Calculate scale to fit both nodes with padding (0.6 factor for more context)
            let scale = 0.6 / Math.max(dx / width, dy / height);
            scale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, scale)); // Clamp scale

            // Calculate transform to center the midpoint
            const transform = d3.zoomIdentity
                .translate(width / 2, height / 2)
                .scale(scale)
                .translate(-cx, -cy);

            svg.transition().duration(750).call(zoom.transform, transform);
        }

        // --- Helper: Create Expandable Relation Card ---
        function createRelationCard(d, initialExpanded = false) {
            const container = document.createElement("div");
            container.style.border = "1px solid #eee";
            container.style.borderRadius = "4px";
            container.style.marginBottom = "6px";
            container.style.backgroundColor = "#fff";
            container.style.overflow = "hidden";
            container.className = "relation-card";

            // Header
            const header = document.createElement("div");
            header.style.padding = "8px";
            header.style.cursor = "pointer";
            header.style.backgroundColor = "#f9f9f9";
            header.style.display = "flex";
            header.style.flexDirection = "column";
            header.title = "Click to expand/collapse and zoom";
            
            header.innerHTML = `
                 <div style="font-size: 0.95em; color:#333;">
                    <span style="font-size:1.2em; color:${themeColorMap[d.theme]}">●</span>&nbsp;
                    <b>${d.source.id}</b> <span style="color:#888;">→ ${d.relation} →</span> <b>${d.target.id}</b>
                </div>
            `;

            // Body
            const body = document.createElement("div");
            body.style.display = initialExpanded ? "block" : "none";
            body.style.padding = "10px";
            body.style.borderTop = "1px solid #eee";
            body.style.backgroundColor = "#fff";

            // Populate Body (Evidence)
            let html = "";
            html += `<div style="margin-bottom:8px; font-size:0.9em; color:#888;">Theme: ${d.theme} | Type: ${d.source_type}</div>`;
            
            const evidence = linkDetailsMap.get(d.id) || [];

            if (evidence.length === 0) {
                html += `<i>No evidence available.</i>`;
            } else {
                evidence.forEach((evidence_item) => {
                    if (!evidence_item) return;

                    const doi = evidence_item.evidence_doi || evidence_item.doi || evidence_item.DOI || (typeof evidence_item === 'string' ? evidence_item : '');
                    const text = evidence_item.evidence_text || evidence_item.text || evidence_item.Text || "";
                    const m = articlesMetadata[doi] || {};

                    // Evidence text
                    if (text) {
                        html += `<div style="margin-bottom:4px;"><b>"${text}"</b></div>`;
                    }

                    // Citation
                    let authorsLine = "";
                    if (m.authors && m.authors.length) {
                        authorsLine = m.authors.join(", ");
                    }

                    let journalLine = "";
                    if (m.journal) journalLine += m.journal;
                    if (m.published || m.year) {
                        if (journalLine) journalLine += ", ";
                        journalLine += m.published || m.year;
                    }
                    if (doi) {
                        journalLine += ` <a href="https://doi.org/${doi}" target="_blank" style="margin-left:5px;">DOI</a>`;
                    }
                    const page = evidence_item.evidence_page;
                    if (page) {
                        journalLine += ` (Page ${page})`;
                    }

                    html += `<div style="font-size:0.9em; color:#666;">`;

                    if (m.title) {
                        html += `<div><i>${m.title}</i></div>`;
                    }

                    if (authorsLine) {
                        html += `<div>${authorsLine}</div>`;  // primera línea: autores
                    }
                    if (journalLine) {
                        html += `<div>${journalLine}</div>`;  // segunda línea: journal, año, DOI
                    }

                    html += `</div>`;  // cierre del container de cita
                });
            }
            body.innerHTML = html;

            // Interaction
            header.onclick = (e) => {
                e.stopPropagation();
                const isHidden = body.style.display === "none";
                body.style.display = isHidden ? "block" : "none";
                if (isHidden) {
                    // 1. Reset previous highlights in graph
                    g.selectAll(".link")
                        .classed("highlighted", false)
                        .attr("marker-end", l => `url(#arrow_${slugify(l.theme)})`);
                    
                    // 2. Highlight this link specifically
                    link.filter(l => l === d)
                        .classed("highlighted", true)
                        .attr("marker-end", `url(#arrow_highlighted_${slugify(d.theme)})`);

                    focusOnLink(d, true); // Force focus
                    zoomToLink(d);
                }
                else {
                    unfocusAll(); // If the card is being collapsed, try to restore the previous view (path or full graph)
                }
            };

            container.appendChild(header);
            container.appendChild(body);
            return container;
        }

        function showLinkDetails(d) {
            const detailsPanel = document.getElementById("details");
            detailsPanel.innerHTML = '';
            // Show single card expanded
            detailsPanel.appendChild(createRelationCard(d, true));
        }

        function showNodeDetails(n) {
            const detailsPanel = document.getElementById("details");
            detailsPanel.innerHTML = ''; // Clear content

            if (topNodesState.active) {
                const headerDiv = document.createElement("div");
                const title = document.createElement("span");
                title.innerHTML = `<b>Top ${topNodesState.nodes.length} Nodes:</b> `;
                headerDiv.appendChild(title);

                topNodesState.nodes.forEach((node, i) => {
                    if (i > 0) headerDiv.appendChild(document.createTextNode(" | "));
                    const link = document.createElement("a");
                    link.textContent = node.id;

                    if (node.id === n.id) {
                        // Highlight selected node and allow deselecting
                        link.style.fontWeight = "bold";
                        link.style.color = "#333";
                        link.style.cursor = "pointer";
                        link.title = "Click to deselect (show all top nodes)";
                        link.onclick = (e) => {
                            e.stopPropagation();
                            focusTopNodes(topNodesState.nodes.length);
                        };
                    } else {
                        link.style.cursor = "pointer";
                        link.style.color = "#007bff";
                        link.style.textDecoration = "underline";
                        link.onclick = (e) => {
                            e.stopPropagation();
                            focusOnNode(node);
                            showNodeDetails(node);
                        };
                    }
                    headerDiv.appendChild(link);
                });
                headerDiv.appendChild(document.createElement("hr"));
                detailsPanel.appendChild(headerDiv);
            }

            const header = document.createElement("div");
            header.innerHTML = `<b>Node:</b> ${n.id}<br><b>Theme:</b> ${n.theme}<hr><b>Relations:</b><br>`;
            detailsPanel.appendChild(header);

            const out = links.filter(l => l.source.id === n.id || l.target.id === n.id);
            
            const listContainer = document.createElement("div");
            listContainer.style.display = "flex";
            listContainer.style.flexDirection = "column";
            listContainer.style.gap = "2px";

            out.forEach((l) => {
                listContainer.appendChild(createRelationCard(l, false));
            });
            detailsPanel.appendChild(listContainer);
        }

        // --- Level of Detail (LOD) Logic ---
        function updateLOD(t) {
            if (!t) t = d3.zoomTransform(svg.node());
            const k = t.k;
            const width = +svg.attr("width");
            const height = +svg.attr("height");

            // Calculate viewport bounds in graph coordinates
            // screen_x = x * k + t_x  => x = (screen_x - t_x) / k
            const x0 = -t.x / k;
            const x1 = (width - t.x) / k;
            const y0 = -t.y / k;
            const y1 = (height - t.y) / k;
            
            // Buffer to avoid popping artifacts at edges
            const buffer = 100 / k; 

            const isMicro = k >= LOD_THRESHOLD;

            const checkVisibility = (d) => {
                // 1. Logical Filter (Search/Theme) - must be true first
                if (!d.isLogicallyVisible) return "none";

                // 2. Focus Mode Override (Exceptions)
                if (focusMode) {
                    if (!focusedElements.links.has(d)) return "none";
                    if (!focusRespectsLOD) return null;

                    // Si estamos respetando LOD (Vista Comunidad), ocultar arcos si estamos en la vista general
                    // Solo mostrarlos si el usuario hace zoom intencional hacia adentro (1.2x del zoom base)
                    if (k < focusBaseScale * 1.2) return "none";
                }

                // 3. Macro View (Zoomed out) - Hide all links
                if (!isMicro) return "none";

                // 4. Micro View - Spatial Culling
                // Show link ONLY if BOTH source and target are within the viewport
                const s = d.source;
                const t_node = d.target;
                const sIn = s.x >= x0 - buffer && s.x <= x1 + buffer && s.y >= y0 - buffer && s.y <= y1 + buffer;
                const tIn = t_node.x >= x0 - buffer && t_node.x <= x1 + buffer && t_node.y >= y0 - buffer && t_node.y <= y1 + buffer;
                
                return (sIn && tIn) ? null : "none";
            };

            link.style("display", checkVisibility);
            linkLabels.style("display", checkVisibility);
        }
        
        // Exponer updateLOD para que handleGlobalResize pueda usarla
        window.updateLOD = updateLOD;

        // ===== FILTERS =====
        const themesDiv = d3.select("#themes");

        // --- Relation Filter ---
        const filterHeader = themesDiv.append("div")
            .style("cursor", "pointer")
            .style("display", "flex")
            .style("align-items", "center")
            .style("margin-bottom", "5px");

        const filterToggleIcon = filterHeader.append("span")
            .style("margin-right", "4px")
            .html('<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"/></svg>'); // Down arrow

        filterHeader.append("strong").text("Filters");

        filterHeader.append("span")
            .text("(?)")
            .style("font-size", "11px")
            .style("cursor", "help")
            .style("color", "#007bff")
            .style("font-weight", "normal")
            .style("margin-left", "4px")
            .attr("title", 'Use "quotes" for exact match.')
            .on("click", (event) => event.stopPropagation()); // Prevent toggle when clicking help

        const filterControls = themesDiv.append("div").attr("id", "filter-controls").style("display", "flex");

        filterHeader.on("click", function () {
            const isVisible = filterControls.style("display") !== "none";
            filterControls.style("display", isVisible ? "none" : "flex");
            filterToggleIcon.html(isVisible ? '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M4.646 1.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1 0 .708l-6 6a.5.5 0 0 1-.708-.708L10.293 8 4.646 2.354a.5.5 0 0 1 0-.708z"/></svg>' : '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"/></svg>');
        });

        const searchFields = ['All Fields', 'Node', 'Relation', 'Title', 'Author', 'Year'];
        const searchFieldSelect = filterControls.append("select")
            .attr("id", "search-field-select")
            .style("padding", "5px")
            .on("change", updateFilters);

        searchFieldSelect.selectAll("option")
            .data(searchFields)
            .join("option")
            .attr("value", d => d.toLowerCase().replace(' ', ''))
            .text(d => d);

        const filterInput = filterControls.append("input")
            .attr("id", "filter-input")
            .attr("type", "text")
            .style("flex", "1")
            .style("min-width", "0")        // muy importante para flexbox
            .attr("placeholder", "Filter by node, relation, title, author, year...")
            .on("input", debounce(updateFilters, 300));

        filterControls.append("button")
            .attr("class", "ui-button")
            .style("flex", "0 0 32px")      // ancho fijo del botón
            .style("min-width", "32px")     // asegurar ancho mínimo
            .style("padding", "0")          // sin padding extra
            .html("&#128269;") // Search Icon
            .attr("title", "Search")
            .on("click", () => {
                if (focusMode) unfocusAll();
                updateFilters();
            });

        // --- Source Type Filter (Dropdown) ---
        const sourceTypes = ['All', 'Articles', 'Abstracts'];
        const sourceTypeSelect = filterControls.append("select")
            .attr("id", "source-type-select")
            .on("change", updateFilters);

        sourceTypeSelect.selectAll("option")
            .data(sourceTypes)
            .join("option")
            .attr("value", d => d)
            .text(d => d);

        const clearButton = filterControls.append("button")
            .attr("class", "filter-button ui-button secondary")
            .style("width", "auto")
            .text("Clear")
            .on("click", () => {
                filterInput.property("value", "");
                sourceTypeSelect.property("value", "All");
                searchFieldSelect.property("value", "allfields");
                updateFilters();
            });

        // --- Theme Filter ---
        themesDiv.append("hr");
        const themeToggleDiv = themesDiv.append("div");
        themeToggleDiv.append("input")
            .attr("type", "checkbox")
            .attr("id", "toggleAll")
            .property("checked", true)
            .on("change", () => {
                const masterChecked = d3.select("#toggleAll").property("checked");
                d3.selectAll(".themeCheckbox").property("checked", masterChecked);
                updateFilters();
            });
        themeToggleDiv.append("span")
            .attr("id", "toggleAllLabel")
            .text(" Select / Deselect All Themes") // The text will be updated dynamically
            .style("margin-left", "5px");
        themesDiv.append("hr");

        themes.forEach(th => {
            const row = themesDiv.append("div").attr("class", "theme-item");
            row.append("input")
                .attr("type", "checkbox")
                .attr("class", "themeCheckbox")
                .property("checked", true)
                .attr("value", th)
                .on("change", updateFilters);
            row.append("span").attr("class", "theme-color").style("background-color", themeColorMap[th]);
            // Add a span with a unique ID for each theme counter
            row.append("span").attr("id", `theme-label-${slugify(th)}`).text(th).style("font-size", "13px");
        });

        const minimapPanel = d3.select("#minimap-panel");
        minimapPanel.select("#reset-view-btn").remove();
        minimapPanel.append("button")
            .attr("id", "reset-view-btn")
            .attr("class", "ui-button secondary")
            .style("margin-top", "10px")
            .html('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M4.25 5.5a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 00.75-.75v-4a.75.75 0 011.5 0v4A2.25 2.25 0 0112.75 17h-8.5A2.25 2.25 0 012 14.75v-8.5A2.25 2.25 0 014.25 4h4a.75.75 0 010 1.5h-4z" clip-rule="evenodd" /><path fill-rule="evenodd" d="M6.25 10a.75.75 0 00.75.75h8.5a.75.75 0 00.75-.75v-8.5a.75.75 0 00-.75-.75h-8.5a.75.75 0 00-.75.75v8.5zM7.75 3a.75.75 0 01.75-.75h6.5a.75.75 0 01.75.75v6.5a.75.75 0 01-.75.75h-6.5a.75.75 0 01-.75-.75V3z" clip-rule="evenodd" /></svg> Fit to Screen')
            .on("click", resetView);

        const layoutPanel = d3.select("#layout-panel");
        layoutPanel.selectAll("*").remove(); // Clean up previous controls

        // --- Layout Controls ---
        const layoutControls = layoutPanel.append("div").attr("id", "layout-controls");

        // Radio buttons for Layout
        const radioGroup = layoutControls.append("div").attr("class", "control-group").style("margin-bottom", "5px");
        radioGroup.append("span").attr("class", "control-group-header").text("Layout Mode");

        const labelTheme = radioGroup.append("label").style("font-size", "13px").style("display", "block").style("margin-bottom", "4px").style("cursor", "pointer");
        labelTheme.append("input").attr("type", "radio").attr("name", "layout-mode").attr("value", "cluster").property("checked", true);
        labelTheme.append("span").text(" By Theme");

        const labelForce = radioGroup.append("label").style("font-size", "13px").style("display", "block").style("cursor", "pointer");
        labelForce.append("input").attr("type", "radio").attr("name", "layout-mode").attr("value", "force");
        labelForce.append("span").text(" By Component");

        // --- Connection Settings Container (Hidden by default) ---
        const connectionSettings = layoutControls.append("div").attr("id", "connection-settings").style("margin-left", "20px").style("display", "none");

        // 1. Show Areas Checkbox (Moved to top)
        const areaDiv = connectionSettings.append("div");
        const labelArea = areaDiv.append("label").style("font-size", "11px").style("cursor", "pointer").style("color", "#666");
        const checkArea = labelArea.append("input").attr("type", "checkbox").attr("id", "show-areas").property("checked", false);
        labelArea.append("span").text(" Show Areas");

        // Container for Area Sub-options
        const areaSubOptions = connectionSettings.append("div").attr("id", "area-sub-options").style("margin-left", "16px").style("display", "none");

        // 1.1 Show Labels Checkbox
        const labelDiv = connectionSettings.append("div").style("margin-top", "2px");
        const labelLabel = labelDiv.append("label").style("font-size", "11px").style("cursor", "pointer").style("color", "#666");
        const checkLabels = labelLabel.append("input").attr("type", "checkbox").attr("id", "show-labels").property("checked", false);
        labelLabel.append("span").text(" Show Labels");

        // Container for Label Sub-options
        const labelSubOptions = connectionSettings.append("div").attr("id", "label-sub-options").style("margin-left", "16px").style("display", "none");

        // Size Input
        const sizeDiv = labelSubOptions.append("div").style("margin-top", "2px").style("display", "flex").style("align-items", "center");
        sizeDiv.append("span").style("font-size", "11px").style("color", "#666").style("margin-right", "4px").text("Size:");
        const labelSizeInput = sizeDiv.append("input").attr("type", "number").attr("id", "label-size-input").attr("min", 10).attr("max", 100).attr("value", communityLabelFontSize).style("width", "35px").style("font-size", "11px").style("border", "1px solid #ddd").style("border-radius", "3px").style("padding", "1px 2px");

        labelSizeInput.on("change", function () {
            let val = +this.value;
            if (val < 10) val = 10;
            if (val > 100) val = 100;
            this.value = val;
            communityLabelFontSize = val;
            communityLabelLayer.selectAll("text").style("font-size", communityLabelFontSize + "px");
        });

        // Show #Nodes Checkbox
        const countDiv = labelSubOptions.append("div").style("margin-top", "2px");
        const labelCount = countDiv.append("label").style("font-size", "11px").style("cursor", "pointer").style("color", "#666");

        const checkCount = labelCount.append("input").attr("type", "checkbox").attr("id", "show-node-count");
        labelCount.append("span").text(" Show Node Count");

        checkCount.on("change", function () {
            updateLabelText();
        });

        // 1.2 Top N Areas Filter (Nested)
        const topNContainer = connectionSettings.append("div").attr("id", "top-n-container").style("margin-top", "2px").style("display", "flex").style("align-items", "center");
        const checkTopN = topNContainer.append("input").attr("type", "checkbox").attr("id", "limit-top-n-areas");
        topNContainer.append("label").attr("for", "limit-top-n-areas").style("font-size", "11px").style("color", "#666").style("margin-left", "4px").style("margin-right", "4px").text("Show Top");
        const inputTopN = topNContainer.append("input").attr("type", "number").attr("id", "top-n-areas-input").attr("value", 10).attr("min", 1).style("width", "35px").style("font-size", "11px").style("border", "1px solid #ddd").style("border-radius", "3px").style("padding", "1px 2px");
        const maxAreasLabel = topNContainer.append("span").attr("id", "top-n-max-label").style("font-size", "11px").style("color", "#666").style("margin-left", "4px").text("/ ?");

        // Helper to toggle Top N visibility
        function updateTopNVisibility() {
            topNContainer.style("display", "flex");
        }
        // Initialize visibility
        updateTopNVisibility();

        checkArea.on("change", function () {
            const isChecked = this.checked;
            areaSubOptions.style("display", isChecked ? "block" : "none");

            if (!isChecked) {
                // If areas are hidden, ensure nodes are visible to avoid blank screen
                d3.select("#hide-nodes").property("checked", false);
            } else {
                // Automatically check "Hide Nodes" when "Areas" is checked
                d3.select("#hide-nodes").property("checked", true);

                // Automatically check "Labels" when "By Component" is selected
                d3.select("#show-labels").property("checked", true);

                // Automatically check "Show #Nodes" when "Areas" is checked
                d3.select("#show-node-count").property("checked", true);
                labelSubOptions.style("display", "block");

                // Ensure the view is updated based on the new state of "Hide Nodes"
                const isMacro = d3.select("#hide-nodes").property("checked");
                updateMacroView(isMacro);
            }

            updateTopNVisibility();
            if (componentsVisible) updateComponentLayout(true);
        });

        checkLabels.on("change", function () {
            updateTopNVisibility();
            labelSubOptions.style("display", this.checked ? "block" : "none");

            if (componentsVisible) updateComponentLayout(true);
        });

        checkTopN.on("change", () => { if (componentsVisible) updateComponentLayout(true); });
        inputTopN.on("change", function () {
            let val = parseInt(this.value);
            const max = parseInt(d3.select(this).attr("max")) || 9999;
            if (val < 1) val = 1;
            if (val > max) val = max;
            this.value = val;
            if (componentsVisible) updateComponentLayout(true);
        });

        // 2. Hide Nodes Checkbox (Moved inside Area Options)
        const macroDiv = areaSubOptions.append("div").style("margin-top", "2px");
        const labelMacro = macroDiv.append("label").style("font-size", "11px").style("cursor", "pointer").style("color", "#666");
        const checkNodes = labelMacro.append("input").attr("type", "checkbox").attr("id", "hide-nodes").property("checked", false); // Start unchecked (visible)
        labelMacro.append("span").text(" Hide Nodes");

        checkNodes.on("change", function () {
            if (componentsVisible) updateComponentLayout(true); else updateMacroView(this.checked);
        });

        // --- Repulsion Slider --- (Now part of layoutControls)
        const repulsionControl = layoutControls.append("div")
            .style("padding", "5px 10px 10px 10px");

        const repulsionHeader = repulsionControl.append("div").style("display", "flex").style("justify-content", "space-between").style("margin-bottom", "4px").style("align-items", "center");
        repulsionHeader.append("label").attr("for", "repulsion-input").style("font-size", "13px").text("Node Spacing");

        repulsionHeader.append("input").attr("type", "number").attr("id", "repulsion-input").attr("min", 0).attr("max", 20000).attr("step", 100).attr("value", -CHARGE_STRENGTH).style("width", "50px").style("font-size", "11px").style("border", "1px solid #ddd").on("change", function () {
            simulation.force("charge").strength(-this.value);
            simulation.alpha(0.1).restart();
            ticked();
        });

        // --- Node Size Slider ---
        const nodeSizeControl = layoutControls.append("div")
            .style("padding", "0px 10px 10px 10px");

        const nodeSizeHeader = nodeSizeControl.append("div").style("display", "flex").style("justify-content", "space-between").style("margin-bottom", "4px").style("align-items", "center");
        nodeSizeHeader.append("label").attr("for", "node-size-input").style("font-size", "13px").text("Node Size");

        const nodeSizeInputContainer = nodeSizeHeader.append("div").style("display", "flex").style("align-items", "center");
        nodeSizeInputContainer.append("input").attr("type", "number").attr("id", "node-size-input").attr("min", 50).attr("max", 250).attr("step", 10).attr("value", 100).style("width", "40px").style("font-size", "11px").style("border", "1px solid #ddd").on("change", function () {
            const scale = this.value / 100.0;
            radiusScale.range([NODE_RADIUS * scale, (NODE_RADIUS * NODE_RADIUS_MAX_MULTIPLIER) * scale]);
            nodeG.selectAll("circle").attr("r", d => radiusScale(d.degree));
            nodeG.selectAll("text")
                .attr("dy", d => radiusScale(d.degree) + 2)
                .style("font-size", d => Math.max(10, radiusScale(d.degree) * 0.6) + "px");
            // simulation.alpha(0.1).restart();
            ticked();
        });
        nodeSizeInputContainer.append("span").style("font-size", "11px").style("margin-left", "3px").text("%");

        // --- Show Top N Nodes ---
        const topNodesControl = layoutControls.append("div")
            .style("padding", "0px 10px 10px 10px");

        const topNodesHeader = topNodesControl.append("div").style("display", "flex").style("align-items", "center");
        
        const checkTopNodes = topNodesHeader.append("input").attr("type", "checkbox").attr("id", "show-top-nodes");
        topNodesHeader.append("label").attr("for", "show-top-nodes").style("font-size", "13px").style("margin-left", "4px").style("cursor", "pointer").text("Top Nodes");
        
        const inputTopNodesN = topNodesHeader.append("input")
            .attr("type", "number")
            .attr("min", 1)
            .attr("max", 50)
            .attr("value", 5)
            .style("width", "40px")
            .style("margin-left", "8px")
            .style("font-size", "11px")
            .style("border", "1px solid #ddd")
            .on("click", function(event) { event.stopPropagation(); })
            .on("change", function() {
                if (checkTopNodes.property("checked")) {
                    focusTopNodes(parseInt(this.value));
                }
            });

        checkTopNodes.on("change", function() {
            if (this.checked) {
                focusTopNodes(parseInt(inputTopNodesN.property("value")));
            } else {
                unfocusAll();
            }
        });

        // --- Freeze Layout Toggle (Subtle) ---
        const freezeControl = layoutControls.append("div")
            // .style("padding", "10px 10px 5px 10px")
            // .style("margin-top", "10px");

        const freezeLabel = freezeControl.append("label")
            .style("font-size", "13px")
            .style("cursor", "pointer")
            .style("color", "#007bff")
            .style("display", "flex")
            .style("align-items", "center")
            .style("gap", "4px");

        const freezeText = freezeLabel.append("span").text("🛑 Node movement");
        freezeText.append("span")
            .text(" (?)")
            .style("font-size", "11x")
            .style("cursor", "help")
            .style("color", "#007bff")
            .style("font-weight", "normal")
            .attr("title", "Move/stop the movement of the nodes.");

        let isFrozen = false;
        freezeLabel.on("click", () => {
            if (!simulation) return;
            isFrozen = !isFrozen;
            if (isFrozen) {
                simulation.stop();
            } else {
                simulation.alpha(0.3).restart();
            }
        });

        d3.selectAll("input[name='layout-mode']").on("change", function () {
            const selectedLayout = this.value;
            if (selectedLayout === 'cluster') {
                // --- By Theme Mode ---             
                d3.select("#connection-settings").style("display", "none");
                if (componentsVisible) {
                    updateComponentLayout(false); // Turn off component view
                }
                if (window.rescaleMinimap) setTimeout(window.rescaleMinimap, 1200);

                simulation.force("clusterX", d3.forceX(d => getWeightedClusterPos(d, 'x')).strength(CLUSTER_STRENGTH));
                simulation.force("clusterY", d3.forceY(d => getWeightedClusterPos(d, 'y')).strength(CLUSTER_STRENGTH));
            } else { // 'force' layout
                // --- By Component Mode ---
                d3.select("#theme-settings").style("display", "none");
                d3.select("#connection-settings").style("display", "block");
                updateComponentLayout(true); // Turn on
            }

            // Restaurar enfriamiento lento para el reordenamiento global
            simulation.alphaDecay(0.05);
            simulation.alpha(1).restart(); // Reheat the simulation
            resetView();
        });

        // --- Pathfinder Controls (collapsible) ---
        const pathfinderPanel = d3.select("#pathfinder-panel");
        pathfinderPanel.selectAll("*").remove(); // Clean up

        const pathControls = pathfinderPanel.append("div");

        // Search Mode Selector
        const modeGroup = pathControls.append("div").attr("class", "control-group");
        const modeHeader = modeGroup.append("span").attr("class", "control-group-header")
            .style("display", "flex").style("align-items", "center").style("gap", "4px");
        modeHeader.append("span").text("Search Mode");
        modeHeader.append("span")
            .text("(?)")
            .style("font-size", "11px")
            .style("cursor", "help")
            .style("color", "#007bff")
            .style("font-weight", "normal")
            .attr("title", "Shortest: Finds the most direct connection.\nAlternatives: Finds multiple independent paths (up to 10).");

        const modeSelect = modeGroup.append("select").attr("class", "path-mode-select").attr("id", "path-mode-select");

        modeSelect.append("option").attr("value", "shortest").text("Shortest Path");
        modeSelect.append("option").attr("value", "disjoint").text("Alternative Paths");

        const groupA = pathControls.append("div").attr("class", "control-group");
        groupA.append("span").attr("class", "control-group-header").text("Origin Node");
        const containerA = groupA.append("div").attr("class", "path-input-container");
        const inputA = containerA.append("input").attr("class", "path-input").attr("id", "path-input-a").attr("readonly", true).attr("placeholder", "Origin node...");
        const btnA = containerA.append("button").attr("class", "path-select-btn").attr("id", "btn-select-a").html('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path d="M10 3a.75.75 0 01.75.75v1.5h1.5a.75.75 0 010 1.5h-1.5v1.5a.75.75 0 01-1.5 0v-1.5h-1.5a.75.75 0 010-1.5h1.5v-1.5A.75.75 0 0110 3z" /><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.25 5.25a.75.75 0 00-1.5 0v1.5h-1.5a.75.75 0 000 1.5h1.5v1.5a.75.75 0 001.5 0v-1.5h1.5a.75.75 0 000-1.5h-1.5v-1.5z" clip-rule="evenodd" /></svg>').attr("title", "Click to select node from graph");

        // Input B (Compact)
        const groupB = pathControls.append("div").attr("class", "control-group");
        groupB.append("span").attr("class", "control-group-header").text("Destination Node");
        const containerB = groupB.append("div").attr("class", "path-input-container");
        const inputB = containerB.append("input").attr("class", "path-input").attr("id", "path-input-b").attr("readonly", true).attr("placeholder", "Destination node...");
        const btnB = containerB.append("button").attr("class", "path-select-btn").attr("id", "btn-select-b").html('<svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 16 16"><path d="M8.5.5a.5.5 0 0 0-1 0v.518A7 7 0 0 0 1.018 7.5H.5a.5.5 0 0 0 0 1h.518A7 7 0 0 0 7.5 14.982v.518a.5.5 0 0 0 1 0v-.518A7 7 0 0 0 14.982 8.5h.518a.5.5 0 0 0 0-1h-.518A7 7 0 0 0 8.5 1.018V.5Zm-6.48 7A6 6 0 0 1 7.5 2.02v.48a.5.5 0 0 0 1 0v-.48a6 6 0 0 1 5.48 5.48h-.48a.5.5 0 0 0 0 1h.48a6 6 0 0 1-5.48 5.48v-.48a.5.5 0 0 0-1 0v.48a6 6 0 0 1-5.48-5.48h.48a.5.5 0 0 0 0-1h-.48ZM8 10a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z"/></svg>').attr("title", "Click to select node from graph");

        // Container for Alternative Paths options (initially hidden)
        const altPathsOptions = pathControls.append("div")
            .attr("id", "alt-paths-options")
            .attr("class", "control-group")
            .style("display", "none"); // Hidden by default

        const altPathsContainer = altPathsOptions.append("div")
            .style("display", "flex")
            .style("align-items", "center")
            .style("gap", "8px");

        altPathsContainer.append("span")
            .style("font-size", "13px") // Match header font size, but not bold
            .text("Number of Paths:");

        altPathsContainer.append("input")
            .attr("type", "number")
            .attr("id", "alt-paths-input")
            .attr("min", 1)
            .attr("value", 5)
            .style("width", "40px")
            .style("font-size", "12px");

        // Action Buttons
        const actionGroup = pathControls.append("div").style("display", "flex").style("gap", "5px").style("margin-top", "10px");

        actionGroup.append("button")
            .attr("class", "ui-button")
            .text("Find Path")
            .style("flex", "2")
            .on("click", runPathfinding);

        actionGroup.append("button")
            .attr("class", "ui-button secondary")
            .text("Clear")
            .style("flex", "1")
            .on("click", clearPathfinding);

        // Show/hide the options based on the selected mode
        modeSelect.on("change", function() {
            d3.select("#alt-paths-options").style("display", this.value === 'disjoint' ? "block" : "none");
        });

        // Selection Logic
        btnA.on("click", () => togglePathSelection('source'));
        btnB.on("click", () => togglePathSelection('target'));

        function togglePathSelection(type) {
            // If clicking the same button, cancel selection
            if (pathfindingState.selecting === type) {
                pathfindingState.selecting = null;
                d3.selectAll(".path-select-btn").classed("active", false);
                d3.select("#graph").classed("is-selecting", false); // Remove class
                return;
            }

            pathfindingState.selecting = type;
            d3.selectAll(".path-select-btn").classed("active", false);
            d3.select(type === 'source' ? "#btn-select-a" : "#btn-select-b").classed("active", true);
            d3.select("#graph").classed("is-selecting", true); // Add class
        }

        function setPathNode(node, type) {
            pathfindingState[type] = node;
            d3.select(type === 'source' ? "#path-input-a" : "#path-input-b").property("value", node.id);

            // Reset UI
            pathfindingState.selecting = null;
            d3.selectAll(".path-select-btn").classed("active", false);
            d3.select("#graph").classed("is-selecting", false); // Remove class
        }

        function clearPathfinding() {
            pathfindingState.source = null;
            pathfindingState.target = null;
            d3.select("#path-input-a").property("value", "");
            d3.select("#path-input-b").property("value", "");
            d3.select("#graph").classed("is-selecting", false);

            // Reset mode to default
            d3.select("#path-mode-select").property("value", "shortest");
            // Hide alt path options
            d3.select("#alt-paths-options").style("display", "none");

            activePath = null; // Explicitly clear active path so unfocusAll performs full reset
            unfocusAll();
        }

        // --- Community Detection Logic ---
        function updateMacroView(isMacro) {
            if (!componentsVisible || focusMode) return; // Only applies when components are visible and not focused

            const opacity = isMacro ? 0 : 1;
            const pointerEvents = isMacro ? "none" : "auto";

            nodeG.transition().duration(300).style("opacity", opacity).style("pointer-events", pointerEvents);
            nodeG.select("text").transition().duration(300).style("opacity", opacity).style("pointer-events", pointerEvents);
            link.transition().duration(300).style("opacity", isMacro ? 0 : null).style("pointer-events", pointerEvents);
            linkLabels.transition().duration(300).style("opacity", opacity).style("pointer-events", pointerEvents);
        }

        function calculateCommunityCenters(communities, width, height) {
            // Create meta-nodes for each community
            const commNodes = Object.keys(communities).map(cId => {
                const count = communities[cId].length;
                // Estimate radius for collision simulation. Increased to avoid overlap.
                const r = Math.sqrt(count) * (NODE_RADIUS + 5) + 20;
                return { id: cId, r: r, count: count };
            });

            // Sort by count descending (largest first)
            commNodes.sort((a, b) => b.count - a.count);

            if (commNodes.length > 0) {
                // Fix the largest community in the center
                commNodes[0].fx = width / 2;
                commNodes[0].fy = height / 2;
                commNodes[0].x = width / 2;
                commNodes[0].y = height / 2;
            }

            // Spiral parameters (Golden Angle)
            const goldenAngle = Math.PI * (3 - Math.sqrt(5));

            commNodes.forEach((node, i) => {
                if (i === 0) return;

                // Calculate spiral position based on rank (i)
                // Distance grows with sqrt(i) to maintain area density
                const dist = commNodes[0].r + (Math.sqrt(i) * 120);
                const theta = i * goldenAngle;

                node.tx = width / 2 + dist * Math.cos(theta);
                node.ty = height / 2 + dist * Math.sin(theta);
            });

            // Run simulation to resolve overlaps while trying to stay near spiral positions
            const sim = d3.forceSimulation(commNodes)
                .force("x", d3.forceX(d => d.tx || width / 2).strength(0.5))
                .force("y", d3.forceY(d => d.ty || height / 2).strength(0.5))
                // Prevent overlap
                .force("collide", d3.forceCollide(d => d.r + 2).iterations(5)) // Small buffer, more iterations
                .stop();

            // Run enough ticks to stabilize
            for (let i = 0; i < 300; ++i) sim.tick();

            const centers = {};
            commNodes.forEach(c => {
                centers[c.id] = { x: c.x, y: c.y };
            });
            return centers;
        }

        function findConnectedComponents() {
            const componentMap = new Map();
            const visited = new Set();
            let componentId = 0;

            // Build adjacency list once for performance
            const adj = new Map(nodes.map(n => [n.id, []]));
            links.forEach(l => {
                adj.get(l.source.id).push(l.target.id);
                adj.get(l.target.id).push(l.source.id);
            });

            nodes.forEach(node => {
                if (!visited.has(node.id)) {
                    componentId++;
                    const queue = [node.id];
                    visited.add(node.id);

                    while (queue.length > 0) {
                        const uId = queue.shift();
                        componentMap.set(uId, componentId);

                        (adj.get(uId) || []).forEach(vId => {
                            if (!visited.has(vId)) {
                                visited.add(vId);
                                queue.push(vId);
                            }
                        });
                    }
                }
            });
            return componentMap;
        }

        // Expose for global resize handler
        window.updateComponentLayout = updateComponentLayout;

        function updateComponentLayout(shouldBeVisible, forceRestart = false) {
            // If argument is provided use it, otherwise toggle
            const newState = (typeof shouldBeVisible === 'boolean') ? shouldBeVisible : !componentsVisible;

            // Allow re-running if visible (to update settings), but prevent redundant "hide" calls
            if (componentsVisible === newState && !newState) return;
            
            const isSwitchingMode = componentsVisible !== newState;
            componentsVisible = newState;

            if (!componentsVisible) {
                communityHullLayer.selectAll("*").remove();
                communityLabelLayer.selectAll("*").remove();
                window.updateCommunityVisuals = null; // Clear the callback
                topNVisibleNodeIds = null;
                if (!focusMode) {
                    nodeG.style("opacity", 1).style("pointer-events", "auto");
                    nodeG.select("text").style("opacity", 1).style("pointer-events", "auto");
                    link.style("opacity", null).style("pointer-events", "auto");
                    linkLabels.style("opacity", 1).style("pointer-events", "auto");
                    updateFilters();
                }

                // Restore Forces (Theme or Connection based)
                const currentSpacing = parseFloat(d3.select("#repulsion-input").property("value"));
                simulation.force("charge", d3.forceManyBody().strength(-currentSpacing));

                const selectedLayout = d3.select("input[name='layout-mode']:checked").property("value");
                const width = +svg.attr("width");
                const height = +svg.attr("height");

                if (selectedLayout === 'cluster') {
                    simulation.force("clusterX", d3.forceX(d => getWeightedClusterPos(d, 'x')).strength(CLUSTER_STRENGTH));
                    simulation.force("clusterY", d3.forceY(d => getWeightedClusterPos(d, 'y')).strength(CLUSTER_STRENGTH));
                } else {
                    const GENTLE_CENTER_STRENGTH = 0.2;
                    simulation.force("clusterX", d3.forceX(width / 2).strength(GENTLE_CENTER_STRENGTH));
                    simulation.force("clusterY", d3.forceY(height / 2).strength(GENTLE_CENTER_STRENGTH));
                }
                simulation.alpha(0.3).restart();
                if (window.rescaleMinimap) setTimeout(window.rescaleMinimap, 1200);

                return;
            }

            // Show macro option and apply visibility based on its state
            const isMacro = d3.select("#hide-nodes").property("checked");

            if (!focusMode) {
                const opacity = isMacro ? 0 : 1;
                const pointerEvents = isMacro ? "none" : "auto";
                nodeG.style("opacity", opacity).style("pointer-events", pointerEvents);
                nodeG.select("text").style("opacity", opacity).style("pointer-events", pointerEvents);
                link.style("opacity", isMacro ? 0 : null).style("pointer-events", pointerEvents);
                linkLabels.style("opacity", opacity).style("pointer-events", pointerEvents);
            }

            // 1. Run Selected Algorithm
            const componentMap = findConnectedComponents();

            // 2. Group nodes by community
            const communities = {};
            nodes.forEach(n => {
                const cId = componentMap.get(n.id);
                n.community = cId; // Assign community property for force calculation
                if (cId !== undefined) {
                    if (!communities[cId]) communities[cId] = [];
                    communities[cId].push(n);
                }
            });

            // Update Max Count UI
            const totalComponents = Object.keys(communities).length;
            d3.select("#top-n-max-label").text(`/ ${totalComponents}`);
            d3.select("#top-n-areas-input").attr("max", totalComponents);

            // --- Spatial Reorganization: Calculate centers using physics ---
            const width = +svg.attr("width");
            const height = +svg.attr("height");
            const commCenters = calculateCommunityCenters(communities, width, height);

            // Apply Community Forces
            // Stronger pull to community center
            simulation.force("clusterX", d3.forceX(d => commCenters[d.community] ? commCenters[d.community].x : width / 2).strength(0.4));
            simulation.force("clusterY", d3.forceY(d => commCenters[d.community] ? commCenters[d.community].y : width / 2).strength(0.4));

            // Slightly reduce repulsion to allow tighter packing within communities
            // We keep it dynamic based on slider but could adjust if needed
            if (isSwitchingMode || forceRestart) {
                simulation.alphaDecay(0.05); // Restaurar enfriamiento lento para que se acomoden bien
                simulation.alpha(1).restart();
            }

            if (window.rescaleMinimap && (isSwitchingMode || forceRestart)) setTimeout(window.rescaleMinimap, 1200);
            
            // 3. Filter and Draw
            const hulls = [];

            Object.values(communities).forEach(members => {
                // Find "Leader" (highest degree)
                const leader = members.reduce((prev, current) => (prev.degree > current.degree) ? prev : current);

                hulls.push({
                    members: members,
                    leader: leader,
                    color: themeColorMap[leader.theme] // Use leader's theme color for the hull
                });
            });

            // Filter Top N Areas if enabled and Show Areas is active
            let areasData = hulls;
            const limitTopN = d3.select("#limit-top-n-areas").property("checked");
            topNVisibleNodeIds = null; // Reset before calculation

            if (limitTopN) {
                const n = parseInt(d3.select("#top-n-areas-input").property("value")) || 10;
                areasData = [...hulls].sort((a, b) => b.members.length - a.members.length).slice(0, n);

                // Populate the global set for the filter function
                topNVisibleNodeIds = new Set();
                areasData.forEach(h => h.members.forEach(m => topNVisibleNodeIds.add(m.id)));
            }

            // Update visibility and counts based on all active filters, including Top N
            updateFilters();

            // --- Render Hulls based on selected shape ---
            communityHullLayer.selectAll("*").remove(); // Clear previous shapes to switch type
            
            communityHullLayer.selectAll("circle")
                .data(areasData) // Use filtered data for areas
                .join("circle")
                .attr("class", "community-hull")
                .attr("fill", d => d.color)
                // .attr("stroke", d => d.color)
                .attr("stroke", "none")
                // .attr("stroke-width", 0.1)
                .style("fill-opacity", 0.3) // Estilo directo para asegurar exportación correcta
                // .style("stroke-opacity", 0.6);

            // Render Labels
            communityLabelLayer.selectAll("text")
                .data(areasData)
                .join("text")
                .attr("class", "community-label")
                .style("font-size", communityLabelFontSize + "px")
                .attr("text-anchor", "middle")
                // Position is handled in tick via transform
                .call(d3.drag()
                    .on("start", communityDragStarted)
                    .on("drag", communityDragged)
                )
                .on("click", (event, d) => {
                    focusOnCommunity(d);
                    event.stopPropagation();
                });

            // Initial text render
            updateLabelText();

            // Check if areas should be shown
            const showAreas = d3.select("#show-areas").property("checked");
            const showLabels = d3.select("#show-labels").property("checked");
            communityHullLayer.style("display", showAreas ? "inline" : "none");
            communityLabelLayer.style("display", showLabels ? "inline" : "none");

            // Define draw function separately to call it immediately without throttle
            const drawVisuals = () => {
                const showAreas = d3.select("#show-areas").property("checked");
                const showLabels = d3.select("#show-labels").property("checked");

                if (showAreas) {
                    // Optimized calculation loop to avoid d3 overhead in hot path
                    communityHullLayer.selectAll("circle").each(function (d) {
                        const members = d.members;
                        const n = members.length;
                        if (n === 0) return;

                        let sumX = 0, sumY = 0;
                        for (let i = 0; i < n; i++) {
                            sumX += members[i].x;
                            sumY += members[i].y;
                        }
                        const cx = sumX / n;
                        const cy = sumY / n;

                        let maxDistSq = 0;
                        for (let i = 0; i < n; i++) {
                            const dx = members[i].x - cx;
                            const dy = members[i].y - cy;
                            const dSq = dx * dx + dy * dy;
                            if (dSq > maxDistSq) maxDistSq = dSq;
                        }
                        const r = Math.sqrt(maxDistSq) + 20;

                        // Direct DOM manipulation is faster than d3.select(this).attr(...)
                        this.setAttribute("cx", cx);
                        this.setAttribute("cy", cy);
                        this.setAttribute("r", r);
                    });
                }

                if (showLabels) {
                    communityLabelLayer.selectAll("text")
                        .attr("transform", d => `translate(${d.leader.x}, ${d.leader.y - radiusScale(d.leader.degree) - 20})`);
                }
            };

            // Execute immediately (unthrottled) to ensure initial render
            drawVisuals();

            // Attach throttled version to tick for performance during animation
            window.updateCommunityVisuals = throttle(drawVisuals, 100);
        }

        function updateLabelText() {
            const showCount = d3.select("#show-node-count").property("checked");

            communityLabelLayer.selectAll("text").each(function (d) {
                const el = d3.select(this);
                el.text(""); // Clear existing

                // Main Label
                el.append("tspan")
                    .attr("x", 0)
                    .attr("dy", 0)
                    .text(d.leader.id);

                // Count Label (Centered below)
                if (showCount) {
                    el.append("tspan")
                        .attr("x", 0)
                        .attr("dy", "1.1em")
                        .style("font-size", "1em")
                        .text(`(${d.members.length} nodes)`);
                }
            });
        }

        // --- Pathfinding Algorithms ---

        // Helper: Build Adjacency List
        function buildAdjacencyList() {
            const adj = new Map();
            nodes.forEach(n => adj.set(n.id, []));
            links.forEach(l => {
                adj.get(l.source.id).push({ node: l.target, link: l });
                adj.get(l.target.id).push({ node: l.source, link: l });
            });
            return adj;
        }

        // Algorithm 1: Single Shortest Path (BFS)
        function findSingleShortestPath(startNode, endNode, adj) {
            const queue = [[startNode]];
            const visited = new Set([startNode.id]);
            const parentMap = new Map();
            let found = false;

            while (queue.length > 0) {
                const path = queue.shift();
                const node = path[path.length - 1];

                if (node.id === endNode.id) {
                    found = true;
                    break;
                }

                const neighbors = adj.get(node.id) || [];
                for (const neighbor of neighbors) {
                    if (!visited.has(neighbor.node.id)) {
                        visited.add(neighbor.node.id);
                        parentMap.set(neighbor.node.id, { parent: node, link: neighbor.link });
                        queue.push([...path, neighbor.node]);
                    }
                }
            }

            if (!found) return null;

            // Reconstruct
            const pathNodes = [endNode];
            const pathLinks = [];
            let curr = endNode;
            while (curr.id !== startNode.id) {
                const info = parentMap.get(curr.id);
                pathNodes.unshift(info.parent);
                pathLinks.unshift(info.link);
                curr = info.parent;
            }
            return { nodes: pathNodes, links: pathLinks, visitedCount: visited.size };
        }

        // Helper: Dijkstra for Weighted Paths (needed for Penalty Method)
        function findPathDijkstra(startNode, endNode, adj, weights) {
            const dist = new Map();
            const prev = new Map();
            // Simple priority queue (array sorted by cost)
            const pq = [{ node: startNode, cost: 0 }];

            dist.set(startNode.id, 0);

            while (pq.length > 0) {
                pq.sort((a, b) => a.cost - b.cost); // Sort to get min cost
                const { node: u, cost } = pq.shift();

                if (cost > (dist.get(u.id) ?? Infinity)) continue;
                if (u.id === endNode.id) break;

                const neighbors = adj.get(u.id) || [];
                for (const neighbor of neighbors) {
                    const linkWeight = weights.get(neighbor.link) || 1;
                    const newCost = cost + linkWeight;

                    if (newCost < (dist.get(neighbor.node.id) ?? Infinity)) {
                        dist.set(neighbor.node.id, newCost);
                        prev.set(neighbor.node.id, { parent: u, link: neighbor.link });
                        pq.push({ node: neighbor.node, cost: newCost });
                    }
                }
            }

            if (!prev.has(endNode.id)) return null;

            // Reconstruct
            const pathNodes = [endNode];
            const pathLinks = [];
            let curr = endNode;
            while (curr.id !== startNode.id) {
                const info = prev.get(curr.id);
                pathNodes.unshift(info.parent);
                pathLinks.unshift(info.link);
                curr = info.parent;
            }
            return { nodes: pathNodes, links: pathLinks };
        }

        // Algorithm 2: Alternative Paths (Iterative Penalty Method)
        function findAlternativePaths(startNode, endNode, adj, k = 5) {
            const paths = [];
            const pathSignatures = new Set(); // To avoid exact duplicates
            const edgeWeights = new Map(); // link -> weight (default 1)
            const allNodes = new Set();
            const allLinks = new Set();

            for (let i = 0; i < k; i++) {
                const res = findPathDijkstra(startNode, endNode, adj, edgeWeights);
                if (!res) break;

                // Create a unique signature for this path to check duplicates
                // (Using link indices or IDs)
                const sig = res.links.map(l => l.index).join("-");
                if (pathSignatures.has(sig)) {
                    // If we found the exact same path again, increase penalty and retry
                    // without adding it to results, to force finding a new one next loop.
                    res.links.forEach(l => {
                        const w = edgeWeights.get(l) || 1;
                        edgeWeights.set(l, w + 1);
                    });
                    continue;
                }
                pathSignatures.add(sig);

                paths.push(res);
                res.nodes.forEach(n => allNodes.add(n));
                res.links.forEach(l => {
                    allLinks.add(l);
                    // Apply penalty: Increase weight of used edges
                    // This makes them "expensive" for the next iteration, encouraging alternatives
                    const w = edgeWeights.get(l) || 1;
                    edgeWeights.set(l, w + 2.0);
                });
            }

            if (paths.length === 0) return null;
            return { nodes: Array.from(allNodes), links: Array.from(allLinks), paths: paths };
        }

        function runPathfinding() {
            const startNode = pathfindingState.source;
            const endNode = pathfindingState.target;
            const mode = d3.select("#path-mode-select").property("value");

            if (!startNode || !endNode) {
                alert("Please select both an Origin and a Destination node.");
                return;
            }

            if (startNode.id === endNode.id) {
                alert("Origin and Destination are the same!");
                return;
            }

            showLoader('Finding Paths...');
            setTimeout(() => {

            const adj = buildAdjacencyList();
            let result = null;
            let highlights = [];

            // Execute selected algorithm
            if (mode === "shortest") {
                result = findSingleShortestPath(startNode, endNode, adj);
                if (result) {
                    highlights.push({ color: "#FFD700", links: result.links }); // Gold
                }
            } else if (mode === "disjoint") {
                const numPaths = parseInt(d3.select("#alt-paths-input").property("value")) || 5;
                result = findAlternativePaths(startNode, endNode, adj, numPaths);
                if (result) {
                    // Assign different colors to each path
                    const colors = ['#FFD700', '#00BFFF', '#FF69B4', '#32CD32', '#FF4500', '#9400D3', '#FF1493', '#00CED1', '#FF8C00', '#8A2BE2'];
                    result.paths.forEach((p, i) => {
                        highlights.push({ color: colors[i % colors.length], links: p.links });
                    });
                }
            }

            if (result) {
                activePath = { nodes: result.nodes, links: result.links }; // Store for restoration
                focusOnPath(result.nodes, result.links);
                drawPathHighlights(highlights);
                
                // --- Render Path Details ---
                const detailsPanel = document.getElementById("details");
                detailsPanel.innerHTML = '';

                if (mode === "shortest") {
                    const header = document.createElement("div");
                    header.innerHTML = `<b>Shortest Path Found!</b><br>Length: ${result.links.length} hops.<hr>`;
                    detailsPanel.appendChild(header);

                    const listContainer = document.createElement("div");
                    listContainer.style.display = "flex";
                    listContainer.style.flexDirection = "column";
                    listContainer.style.gap = "2px";

                    result.links.forEach(l => {
                        listContainer.appendChild(createRelationCard(l, false));
                    });
                    detailsPanel.appendChild(listContainer);

                } else if (mode === "disjoint") {
                    const header = document.createElement("div");
                    header.innerHTML = `<b>Alternative Paths Found!</b><br>Found ${result.paths.length} independent routes.<hr>`;
                    detailsPanel.appendChild(header);

                    const colors = ['#FFD700', '#00BFFF', '#FF69B4', '#32CD32', '#FF4500', '#9400D3', '#FF1493', '#00CED1', '#FF8C00', '#8A2BE2'];

                    result.paths.forEach((p, i) => {
                        const pathColor = colors[i % colors.length];
                        
                        const pathContainer = document.createElement("div");
                        pathContainer.style.marginBottom = "15px";

                        const pathHeader = document.createElement("div");
                        pathHeader.style.marginBottom = "5px";
                        pathHeader.style.fontWeight = "bold";
                        pathHeader.style.color = "#333";
                        pathHeader.style.borderBottom = `2px solid ${pathColor}`;
                        pathHeader.style.display = "inline-block";
                        pathHeader.innerText = `Path ${i + 1} (${p.links.length} hops)`;
                        pathContainer.appendChild(pathHeader);

                        const listContainer = document.createElement("div");
                        listContainer.style.display = "flex";
                        listContainer.style.flexDirection = "column";
                        listContainer.style.gap = "2px";

                        p.links.forEach(l => {
                            listContainer.appendChild(createRelationCard(l, false));
                        });
                        pathContainer.appendChild(listContainer);
                        detailsPanel.appendChild(pathContainer);
                    });
                }
            } else {
                // Broken Bridge Logic
                // Run a partial BFS to find reachable nodes from Start
                const queue = [startNode];
                const visited = new Set([startNode.id]);
                const parentMap = new Map();

                while (queue.length > 0) {
                    const node = queue.shift();
                    const neighbors = adj.get(node.id) || [];
                    for (const neighbor of neighbors) {
                        if (!visited.has(neighbor.node.id)) {
                            visited.add(neighbor.node.id);
                            parentMap.set(neighbor.node.id, { parent: node, link: neighbor.link });
                            queue.push(neighbor.node);
                        }
                    }
                }

                let closestNode = null;
                let minDist = Infinity;

                visited.forEach(nodeId => {
                    const n = nodes.find(x => x.id === nodeId);
                    const dist = (n.x - endNode.x) ** 2 + (n.y - endNode.y) ** 2;
                    if (dist < minDist) {
                        minDist = dist;
                        closestNode = n;
                    }
                });

                // Reconstruct path to closest node
                const pathNodes = [closestNode];
                const pathLinks = [];
                let curr = closestNode;
                while (curr.id !== startNode.id) {
                    const info = parentMap.get(curr.id);
                    if (!info) break;
                    pathNodes.unshift(info.parent);
                    pathLinks.unshift(info.link);
                    curr = info.parent;
                }

                // Add target to visible nodes so we see the gap
                pathNodes.push(endNode);

                activePath = { nodes: pathNodes, links: pathLinks }; // Store for restoration
                focusOnPath(pathNodes, pathLinks);
                drawPathHighlights([{ color: "#FF4444", links: pathLinks }]); // Red highlight for incomplete path

                document.getElementById("details").innerHTML =
                    `<b>No se encontró una ruta directa.</b><br>Se muestra el camino más cercano al destino.`;
            }
                hideLoader();
            }, 10);
        }

        // --- Export Controls (collapsible) ---
        const exportPanel = d3.select("#export-panel");
        exportPanel.selectAll("*").remove();
        const exportControls = exportPanel.append("div");

        // Download Button
        exportControls.append("button")
            .attr("class", "ui-button")
            .text("Download PNG")
            .style("margin-bottom", "8px")
            .on("click", () => {
                const scale = 4; // Fixed to 300dpi (approx 4x screen resolution)
                const withLegend = exportControls.select('#export-legend').property("checked");
                downloadPNG(scale, "white", withLegend);
            });

        // PNG Options Container
        const pngOptions = exportControls.append("div").style("margin-bottom", "15px").style("padding-left", "5px");

        // Legend Option
        const legendLabel = pngOptions.append("label").style("font-size", "12px").style("cursor", "pointer").style("display", "flex").style("align-items", "center");
        legendLabel.append("input").attr("type", "checkbox").attr("id", "export-legend").property("checked", true).style("margin-right", "5px");
        legendLabel.append("span").text("Include Legend");

        exportControls.append("button")
            .attr("class", "ui-button secondary")
            .text("Download CSV")
            .on("click", downloadCSV);

        function downloadPNG(scale, bg, withLegend) {
            const svgNode = svg.node();
            const width = +svg.attr("width");
            const height = +svg.attr("height");

            // --- HIDE UI ELEMENTS FOR EXPORT ---
            const indicator = svg.select("#unfocus-indicator");
            const originalIndicatorDisplay = indicator.style("display");
            indicator.style("display", "none");
            // ------------------------------------

            const clone = svgNode.cloneNode(true);
            const scaledWidth = width * scale;
            const scaledHeight = height * scale;

            // Set viewBox to original dimensions to ensure scaling works correctly
            clone.setAttribute("viewBox", `0 0 ${width} ${height}`);
            // Explicitly set pixel dimensions to force high-res rasterization
            clone.setAttribute("width", scaledWidth + "px");
            clone.setAttribute("height", scaledHeight + "px");

            // Inject Styles
            const style = document.createElement("style");
            let cssText = "";
            
            // 1. Capture inline <style> tags (like your previous code)
            document.querySelectorAll("style").forEach(tag => {
                cssText += tag.textContent + "\n";
            });

            // 2. Attempt to collect all CSS rules from external stylesheets (styles.css)
            for (let i = 0; i < document.styleSheets.length; i++) {
                try {
                    const rules = document.styleSheets[i].cssRules;
                    if (rules) {
                        for (let j = 0; j < rules.length; j++) {
                            cssText += rules[j].cssText + "\n";
                        }
                    }
                } catch (e) {}
            }
            style.textContent = cssText;
            clone.insertBefore(style, clone.firstChild);

            // Handle Background (add rect behind everything)
            if (bg === "white") {
                const bgRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                bgRect.setAttribute("width", "100%");
                bgRect.setAttribute("height", "100%");
                bgRect.setAttribute("fill", "white");
                clone.insertBefore(bgRect, clone.firstChild);
            }

            // Handle Legend
            if (withLegend) {
                // Calculate dynamic width based on longest theme name (approx 8px per char + padding)
                const maxThemeLen = themes.reduce((max, t) => Math.max(max, t.length), 0);
                const legendWidth = Math.max(150, 30 + (maxThemeLen * 8));
                const itemHeight = 20;
                const padding = 10;
                const titleHeight = 25;
                const legendHeight = (themes.length * itemHeight) + padding * 2 + titleHeight;

                const legendG = document.createElementNS("http://www.w3.org/2000/svg", "g");
                // Position bottom-right with margin
                const x = width - legendWidth - 10;
                const y = height - legendHeight - 10;
                legendG.setAttribute("transform", `translate(${x}, ${y})`);

                // Background
                const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                rect.setAttribute("width", legendWidth);
                rect.setAttribute("height", legendHeight);
                rect.setAttribute("fill", "white");
                rect.setAttribute("stroke", "#ccc");
                rect.setAttribute("stroke-width", "1");
                rect.setAttribute("rx", "4");
                rect.setAttribute("fill-opacity", "0.95");
                legendG.appendChild(rect);

                // Title
                const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
                title.textContent = "Legend";
                title.setAttribute("x", padding);
                title.setAttribute("y", padding + 14);
                title.setAttribute("font-family", "Arial, sans-serif");
                title.setAttribute("font-size", "14px");
                title.setAttribute("font-weight", "bold");
                title.setAttribute("fill", "#333");
                legendG.appendChild(title);

                // Items
                themes.forEach((th, i) => {
                    const itemY = padding + titleHeight + (i * itemHeight) + 10;

                    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                    circle.setAttribute("cx", padding + 6);
                    circle.setAttribute("cy", itemY - 4);
                    circle.setAttribute("r", 5);
                    circle.setAttribute("fill", themeColorMap[th]);
                    legendG.appendChild(circle);

                    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
                    text.textContent = th;
                    text.setAttribute("x", padding + 20);
                    text.setAttribute("y", itemY);
                    text.setAttribute("font-family", "Arial, sans-serif");
                    text.setAttribute("font-size", "12px");
                    text.setAttribute("fill", "#333");
                    legendG.appendChild(text);
                });

                clone.appendChild(legendG);
            }

            const serializer = new XMLSerializer();
            const svgString = serializer.serializeToString(clone);
            const blob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
            const url = URL.createObjectURL(blob);

            const img = new Image();
            // Set image dimensions to match the scaled SVG
            img.width = scaledWidth;
            img.height = scaledHeight;

            img.onload = function () {
                const canvas = document.createElement("canvas");
                canvas.width = scaledWidth;
                canvas.height = scaledHeight;
                const ctx = canvas.getContext("2d");

                // Use high quality smoothing
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = "high";

                ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);

                canvas.toBlob((blob) => {
                    const reader = new FileReader();
                    reader.onload = function (e) {
                        const buffer = e.target.result;
                        const dpiBuffer = injectDpi(buffer, 300); // Inject 300 DPI metadata
                        const newBlob = new Blob([dpiBuffer], { type: "image/png" });
                        const finalUrl = URL.createObjectURL(newBlob);

                        const a = document.createElement("a");
                        a.download = "graph_export.png";
                        a.href = finalUrl;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(finalUrl);

                        // --- RESTORE UI ELEMENTS ---
                        indicator.style("display", originalIndicatorDisplay);
                    };
                    reader.readAsArrayBuffer(blob);
                }, "image/png");
            };
            img.onerror = () => {
                indicator.style("display", originalIndicatorDisplay); // Restore on error
                URL.revokeObjectURL(url);
                alert("An error occurred while exporting the PNG.");
            };
            img.src = url;
        }

        function downloadCSV() {
            const checkedThemes = Array.from(document.querySelectorAll("#themes input.themeCheckbox:checked")).map(i => i.value);
            const activeSourceType = d3.select("#source-type-select").property("value");
            const searchTerm = d3.select("#filter-input").property("value").toLowerCase();

            // Determine allowed nodes if Top N filter is active
            let topNAllowedIds = null;
            if (componentsVisible && !focusMode && d3.select("#limit-top-n-areas").property("checked")) {
                const n = parseInt(d3.select("#top-n-areas-input").property("value")) || 10;
                const componentMap = findConnectedComponents();
                const communities = {};
                nodes.forEach(node => {
                    const cId = componentMap.get(node.id);
                    if (cId !== undefined) {
                        if (!communities[cId]) communities[cId] = [];
                        communities[cId].push(node);
                    }
                });
                // Sort by size (descending) and take top N
                const topCommunities = Object.values(communities)
                    .sort((a, b) => b.length - a.length)
                    .slice(0, n);

                topNAllowedIds = new Set();
                topCommunities.forEach(group => group.forEach(node => topNAllowedIds.add(node.id)));
            }

            const linksToExport = links.filter(d => {
                // 1. Global Filters (Must match what is visible via filters)
                let sourceMatch = true;
                if (activeSourceType === 'Articles' && d.source_type !== 'article') sourceMatch = false;
                if (activeSourceType === 'Abstracts' && d.source_type !== 'abstract') sourceMatch = false;
                if (!sourceMatch) return false;

                if (!checkedThemes.includes(d.theme)) return false;

                // Check Top N Filter
                if (topNAllowedIds) {
                    if (!topNAllowedIds.has(d.source.id) || !topNAllowedIds.has(d.target.id)) return false;
                }

                if (searchTerm !== "") {
                    const basicMatch = d.source.id.toLowerCase().includes(searchTerm) ||
                        d.target.id.toLowerCase().includes(searchTerm) ||
                        d.relation.toLowerCase().includes(searchTerm) ||
                        d.theme.toLowerCase().includes(searchTerm);

                    const evidence = linkDetailsMap.get(d.id) || [];
                    const evidenceMatch = evidence.some(evidenceItem => {
                        const doi = evidenceItem.evidence_doi || evidenceItem.doi || (typeof evidenceItem === 'string' ? evidenceItem : '');
                        const m = articlesMetadata[doi] || {};
                        return (m.title || "").toLowerCase().includes(searchTerm) ||
                            (m.published || "").toString().toLowerCase().includes(searchTerm) ||
                            (Array.isArray(m.authors) ? m.authors.join(' ') : '').toLowerCase().includes(searchTerm);
                    });

                    if (!basicMatch && !evidenceMatch) return false;
                }

                const evidence = linkDetailsMap.get(d.id) || [];

                // 2. Focus Mode (If active, only export focused/expanded links)
                if (focusMode) {
                    if (expansionSeedNodes.size > 0) {
                        // Expansion mode (Ctrl+Click)
                        return expandedNodeIds.has(d.source.id) && expandedNodeIds.has(d.target.id);
                    } else {
                        // Simple focus mode (Single Click)
                        return focusedElements.links.has(d);
                    }
                }

                return true;
            });

            if (linksToExport.length === 0) {
                alert("No relations to export based on current filters/view.");
                return;
            }

            // CSV Header
            let csvContent = "Subject;Relation;Object;Theme;SourceType;DOI;EvidenceText\n";

            linksToExport.forEach(l => {
                const evidence = linkDetailsMap.get(l.id) || [];
                const dois = evidence.map(e => e.evidence_doi || e.doi || (typeof e === 'string' ? e : '')).join("|");
                const evidenceTexts = evidence.map(e => e.evidence_text || e.text || "").join(" | ");

                const row = [
                    l.source.id,
                    l.relation,
                    l.target.id,
                    l.theme,
                    l.source_type,
                    dois,
                    evidenceTexts
                ].map(field => {
                    // Escape quotes and wrap in quotes if necessary (Standard CSV)
                    const str = String(field || "").replace(/"/g, '""');
                    return /[;\n"]/.test(str) ? `"${str}"` : str;
                }).join(";");
                csvContent += row + "\n";
            });

            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.setAttribute("href", url);
            link.setAttribute("download", "graph_relations.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }
        
        function renderGlobalList() {
            const detailsPanel = document.getElementById("details");
            // Only render list if we are NOT in focus mode (i.e. user hasn't clicked a specific node/link yet)
            // Exception: If we are in Top Nodes mode, we DO want to render the list
            if (focusMode && !topNodesState.active) return;

            detailsPanel.innerHTML = '';
            
            const sourceLinks = topNodesState.active ? topNodesState.links : filteredLinks;
            const totalItems = sourceLinks.length;

            if (topNodesState.active) {
                const headerDiv = document.createElement("div");
                const title = document.createElement("span");
                title.innerHTML = `<b>Top ${topNodesState.nodes.length} Nodes:</b> `;
                headerDiv.appendChild(title);

                topNodesState.nodes.forEach((n, i) => {
                    if (i > 0) headerDiv.appendChild(document.createTextNode(" | "));
                    const link = document.createElement("a");
                    link.textContent = n.id;
                    link.style.cursor = "pointer";
                    link.style.color = "#007bff";
                    link.style.textDecoration = "underline";
                    link.onclick = (e) => {
                        e.stopPropagation();
                        focusOnNode(n);
                        showNodeDetails(n);
                    };
                    headerDiv.appendChild(link);
                });
                headerDiv.appendChild(document.createElement("hr"));
                detailsPanel.appendChild(headerDiv);
            }
            
            if (totalItems === 0) {
                const msg = document.createElement("div");
                msg.innerHTML = '<i>No matching relations found.</i>';
                detailsPanel.appendChild(msg);
                return;
            }

            const totalPages = Math.ceil(totalItems / ITEMS_PER_PAGE);
            
            // Ensure currentPage is valid
            if (currentPage > totalPages) currentPage = totalPages;
            if (currentPage < 1) currentPage = 1;

            const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
            const endIndex = Math.min(startIndex + ITEMS_PER_PAGE, totalItems);
            const pageItems = sourceLinks.slice(startIndex, endIndex);

            // --- Pagination Controls (Top) ---
            const paginationDiv = document.createElement("div");
            paginationDiv.style.display = "flex";
            paginationDiv.style.justifyContent = "center";
            paginationDiv.style.alignItems = "center";
            paginationDiv.style.marginBottom = "10px";
            paginationDiv.style.gap = "5px";

            const createBtn = (html, onClick, disabled) => {
                const btn = document.createElement("button");
                btn.className = "ui-button secondary";
                btn.style.padding = "0";
                btn.style.width = "28px";
                btn.style.height = "28px";
                btn.style.minWidth = "auto";
                btn.style.lineHeight = "26px";
                btn.innerHTML = html;
                btn.disabled = disabled;
                btn.onclick = onClick;
                return btn;
            };

            const prevBtn = createBtn("&#9664;", () => { // Left Arrow
                if (currentPage > 1) {
                    currentPage--;
                    renderGlobalList();
                }
            }, currentPage === 1);

            const pageInput = document.createElement("input");
            pageInput.type = "number";
            pageInput.min = 1;
            pageInput.max = totalPages;
            pageInput.value = currentPage;
            pageInput.style.width = "40px";
            pageInput.style.textAlign = "center";
            pageInput.style.fontSize = "12px";
            pageInput.style.padding = "2px";
            pageInput.onchange = function() {
                let val = parseInt(this.value);
                if (isNaN(val)) val = 1;
                if (val < 1) val = 1;
                if (val > totalPages) val = totalPages;
                currentPage = val;
                renderGlobalList();
            };

            const totalLabel = document.createElement("span");
            totalLabel.innerText = `/ ${totalPages}`;
            totalLabel.style.fontSize = "12px";
            totalLabel.style.marginRight = "5px";

            const nextBtn = createBtn("&#9654;", () => { // Right Arrow
                if (currentPage < totalPages) {
                    currentPage++;
                    renderGlobalList();
                }
            }, currentPage === totalPages);

            paginationDiv.appendChild(prevBtn);
            paginationDiv.appendChild(pageInput);
            paginationDiv.appendChild(totalLabel);
            paginationDiv.appendChild(nextBtn);
            
            detailsPanel.appendChild(paginationDiv);

            const countDiv = document.createElement("div");
            countDiv.style.marginBottom = "5px";
            countDiv.style.fontSize = "11px";
            countDiv.style.color = "#555";
            countDiv.style.textAlign = "center";
            countDiv.innerText = `Showing ${startIndex + 1}-${endIndex} of ${totalItems} triplets`;
            detailsPanel.appendChild(countDiv);

            const listContainer = document.createElement("div");
            listContainer.style.display = "flex";
            listContainer.style.flexDirection = "column";
            listContainer.style.gap = "2px";

            pageItems.forEach(l => {
                listContainer.appendChild(createRelationCard(l, false));
            });

            detailsPanel.appendChild(listContainer);
        }

        function updateFilters() {
            const checkedThemes = Array.from(document.querySelectorAll("#themes input.themeCheckbox:checked"))
                .map(i => i.value);

            // Get active source type from the dropdown
            const activeSourceType = d3.select("#source-type-select").property("value");
            const searchField = d3.select("#search-field-select").property("value");
            
            let rawSearchTerm = filterInput.property("value");
            let exactMatch = false;

            // Check for exact search syntax ("...")
            if (rawSearchTerm.startsWith('"') && rawSearchTerm.endsWith('"') && rawSearchTerm.length > 1) {
                exactMatch = true;
                rawSearchTerm = rawSearchTerm.substring(1, rawSearchTerm.length - 1);
            }
            const searchTerm = rawSearchTerm.toLowerCase();
            
            visibleNodeIds.clear();
            filteredLinks = []; // Reset global list

            // Object to keep the count of visible links per theme
            const visibleCounts = {};
            themes.forEach(th => { visibleCounts[th] = 0; });

            function isSourceVisible(d) {
                if (activeSourceType === 'All') return true;
                return (activeSourceType === 'Articles' && d.source_type === 'article') || (activeSourceType === 'Abstracts' && d.source_type === 'abstract');
            }

            function isLinkVisible(d) {
                const sourceMatch = isSourceVisible(d);
                if (!sourceMatch) return false; // Apply source type filter first
                const themeMatch = checkedThemes.includes(d.theme);
                if (!themeMatch) return false; // Always apply theme filter first

                // Check against Top N filter if it's active
                if (componentsVisible && topNVisibleNodeIds && !focusMode) {
                    if (!topNVisibleNodeIds.has(d.source.id) || !topNVisibleNodeIds.has(d.target.id)) {
                        return false;
                    }
                }

                if (searchTerm === "") return true; // If there's no search term, it's visible

                const evidence = linkDetailsMap.get(d.id) || [];

                // Helper for matching based on exactMatch flag
                const match = (text) => {
                    if (!text) return false;
                    const lowerText = String(text).toLowerCase();
                    return exactMatch ? lowerText === searchTerm : lowerText.includes(searchTerm);
                };

                const nodeMatch = () => match(d.source.id) || match(d.target.id);
                const relationMatch = () => match(d.relation);
                const evidenceMatch = (field) => evidence.some(evidenceItem => {
                    const doi = evidenceItem.evidence_doi || evidenceItem.doi || (typeof evidenceItem === 'string' ? evidenceItem : '');
                    const m = articlesMetadata[doi] || {};

                    if (field === 'title') return match(m.title);
                    if (field === 'author') {
                        return Array.isArray(m.authors) && m.authors.some(author => match(author));
                    }
                    if (field === 'year') return match(m.year);

                    // for 'allfields'
                    const authorMatch = Array.isArray(m.authors) ? m.authors.some(author => match(author)) : false;
                    return match(m.title) || authorMatch || match(m.year);
                });

                switch (searchField) {
                    case 'node': return nodeMatch();
                    case 'relation': return relationMatch();
                    case 'title': return evidenceMatch('title');
                    case 'author': return evidenceMatch('author');
                    case 'year': return evidenceMatch('year');
                    case 'allfields':
                    default:
                        return nodeMatch() || relationMatch() || evidenceMatch('allfields') || match(d.theme);
                }
            }

            // Calculate logical visibility but defer DOM update to updateLOD
            links.forEach(d => {
                d.isLogicallyVisible = isLinkVisible(d);
                if (d.isLogicallyVisible) {
                    visibleCounts[d.theme]++;
                    visibleNodeIds.add(d.source.id);
                    visibleNodeIds.add(d.target.id);
                    filteredLinks.push(d);
                }
            });

            // Update node visibility
            nodeG.style("display", d => visibleNodeIds.has(d.id) ? null : "none");

            // Update theme labels with new counts
            themes.forEach(th => {
                const totalCount = links.filter(l => l.theme === th).length;
                d3.select(`#theme-label-${slugify(th)}`).text(`${th} (${visibleCounts[th]}/${totalCount})`);
            });

            // Update the global counter for "Select/Deselect All"
            const totalVisible = Object.values(visibleCounts).reduce((sum, count) => sum + count, 0);
            const totalLinks = links.length;
            d3.select("#toggleAllLabel").text(` Select / Deselect All Themes (${totalVisible}/${totalLinks})`);

            d3.select("#toggleAll").property("checked", (checkedThemes.length === themes.length));
            
            currentPage = 1; // Reset to first page on filter change
            // Render the filtered list in the side panel
            renderGlobalList();
            // Apply visual updates
            updateLOD()
        }

        // Initial filter application
        updateFilters();

        // Add listener for the ESC key to unfocus
        // We use a namespace (.graph) to avoid adding multiple listeners if the graph is reloaded
        d3.select("body").on("keydown.graph", function (event) {
            if (event.key === "Escape") {
                unfocusAll();
                // document.getElementById('details').innerHTML = ... (Handled by updateFilters inside unfocusAll logic via renderGlobalList)
            }
        });

        // ===== MINIMAP =====
        function setupMinimap() {
            document.getElementById('minimap-container').style.display = 'block';
            const minimapContainer = d3.select("#minimap");
            const minimapWidth = minimapContainer.node().clientWidth;
            const minimapHeight = minimapContainer.node().clientHeight;

            const minimapSvg = minimapContainer.append("svg");

            // Scales to map main graph coordinates to the minimap
            let xExtent = d3.extent(nodes, d => d.x);
            let yExtent = d3.extent(nodes, d => d.y);

            const buffer = 500; // Add a buffer so nodes are not on the edge
            xExtent = [xExtent[0] - buffer, xExtent[1] + buffer];
            yExtent = [yExtent[0] - buffer, yExtent[1] + buffer];

            const graphWidth = xExtent[1] - xExtent[0];
            const graphHeight = yExtent[1] - yExtent[0];
            const graphAspectRatio = graphWidth / graphHeight;
            const minimapAspectRatio = minimapWidth / minimapHeight;

            // Adjust the domain to match the minimap's aspect ratio and avoid distortion
            if (graphAspectRatio > minimapAspectRatio) {
                const newHeight = graphWidth / minimapAspectRatio;
                yExtent[0] -= (newHeight - graphHeight) / 2;
                yExtent[1] += (newHeight - graphHeight) / 2;
            } else {
                const newWidth = graphHeight * minimapAspectRatio;
                xExtent[0] -= (newWidth - graphWidth) / 2;
                xExtent[1] += (newWidth - graphWidth) / 2;
            }

            let xScale = d3.scaleLinear().domain(xExtent).range([0, minimapWidth]);
            let yScale = d3.scaleLinear().domain(yExtent).range([0, minimapHeight]);

            const minimapNodes = minimapSvg.append("g")
                .selectAll(".minimap-node")
                .data(nodes, d => d.id)
                .join("circle")
                .attr("class", "minimap-node")
                .attr("cx", d => xScale(d.x))
                .attr("cy", d => yScale(d.y))
                .attr("r", 2);

            const viewbox = minimapSvg.append("rect")
                .attr("class", "minimap-viewbox");

            function rescaleMinimap() {
                if (!nodes || nodes.length === 0) return;
                let newXExtent = d3.extent(nodes, d => d.x);
                let newYExtent = d3.extent(nodes, d => d.y);
                if (!newXExtent[0] || !newYExtent[0]) return;

                newXExtent = [newXExtent[0] - buffer, newXExtent[1] + buffer];
                newYExtent = [newYExtent[0] - buffer, newYExtent[1] + buffer];

                const graphWidth = newXExtent[1] - newXExtent[0];
                const graphHeight = newYExtent[1] - newYExtent[0];
                const graphAspectRatio = graphWidth / graphHeight;

                if (graphAspectRatio > minimapAspectRatio) {
                    const newHeight = graphWidth / minimapAspectRatio;
                    newYExtent[0] -= (newHeight - graphHeight) / 2;
                    newYExtent[1] += (newHeight - graphHeight) / 2;
                } else {
                    const newWidth = graphHeight * minimapAspectRatio;
                    newXExtent[0] -= (newWidth - graphWidth) / 2;
                    newXExtent[1] += (newWidth - graphWidth) / 2;
                }
                xScale.domain(newXExtent);
                yScale.domain(newYExtent);

                minimapNodes
                    .transition().duration(500)
                    .attr("cx", d => xScale(d.x))
                    .attr("cy", d => yScale(d.y));

                const currentTransform = d3.zoomTransform(svg.node());
                if (window.updateMinimapViewbox) window.updateMinimapViewbox(currentTransform);
            }
            window.rescaleMinimap = rescaleMinimap;

            // Function to update minimap nodes (e.g., opacity in focus mode)
            window.updateMinimapNodes = function () {
                const visibleIds = focusMode ? getVisibleNodes().map(n => n.id) : nodes.map(n => n.id);
                const visibleSet = new Set(visibleIds);
                minimapNodes.attr("opacity", d => visibleSet.has(d.id) ? 0.8 : 0.1);
            }

            // Function to update the viewbox rectangle
            window.updateMinimapViewbox = function (transform) {
                const { k, x, y } = transform;
                const mainWidth = svg.attr("width");
                const mainHeight = svg.attr("height");

                // Calculate the viewbox coordinates in the minimap space
                const viewboxX = xScale(-x / k);
                const viewboxY = yScale(-y / k);
                const viewboxWidth = xScale(mainWidth / k) - xScale(0);
                const viewboxHeight = yScale(mainHeight / k) - yScale(0);

                viewbox
                    .attr("x", viewboxX)
                    .attr("y", viewboxY)
                    .attr("width", Math.max(0, viewboxWidth))
                    .attr("height", Math.max(0, viewboxHeight));
            }

            // Dragging the minimap viewbox to pan the main view
            viewbox.call(d3.drag().on("drag", function (event) {
                const currentTransform = d3.zoomTransform(svg.node());
                const { k } = currentTransform;

                // Convert mouse movement in the minimap to panning in the main graph
                const dx = -event.dx / (xScale(1) - xScale(0)) * k;
                const dy = -event.dy / (yScale(1) - yScale(0)) * k;

                // Apply the new offset
                svg.call(zoom.translateBy, dx, dy);
            }));

            // Initialize the minimap view
            const initialTransform = d3.zoomTransform(svg.node());
            window.updateMinimapViewbox(initialTransform);
            window.updateMinimapNodes(); // Initialize node opacity

            // Update minimap scales if the simulation moves a lot
            simulation.on("end.minimap", rescaleMinimap);
        }

        // Wait for the simulation to stabilize a bit before creating the minimap
        setTimeout(setupMinimap, 1000);

        function resetView() {
            const width = +svg.attr("width");
            const height = +svg.attr("height");
            const k = ZOOM_INITIAL_SCALE;
            const tx = width / 2 * (1 - k);
            const ty = height / 2 * (1 - k);

            svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity.translate(tx, ty).scale(k)
            );
        }

        } catch (err) {
            console.error("Error processing graph:", err);
            alert("Error processing graph: " + err.message);
            hideLoader();
        }

    }

    function handleGlobalResize() {
        if (!svg || !simulation) return; // Do nothing if graph is not loaded

        const container = d3.select("#graph");
        const width = container.node().clientWidth;
        const height = container.node().clientHeight;

        // Update SVG size
        svg.attr("width", width).attr("height", height);

        // Update position of the unfocus indicator
        svg.select("#unfocus-indicator")
            .attr("x", width / 2);

        // If communities are visible, re-run toggle to update community centers based on new size
        if (componentsVisible) {
            if (window.updateComponentLayout) window.updateComponentLayout(true, true);
            return;
        }

        // Update cluster center positions
        const margin = 80;
        themes.forEach((th, i) => {
            const x = margin + i * (Math.max(width - 2 * margin, 0)) / Math.max(themes.length - 1, 1);
            clusterCenters[th] = { x: x, y: height / 2 };
        });

        // Update simulation forces based on the current layout mode
        const selectedLayout = d3.select("input[name='layout-mode']:checked").property("value");

        if (selectedLayout === 'cluster') {
            simulation.force("clusterX", d3.forceX(d => getWeightedClusterPos(d, 'x')).strength(CLUSTER_STRENGTH));
            simulation.force("clusterY", d3.forceY(d => getWeightedClusterPos(d, 'y')).strength(CLUSTER_STRENGTH));
        } else { // 'force' layout
            const GENTLE_CENTER_STRENGTH = 0.2;
            simulation.force("clusterX", d3.forceX(width / 2).strength(GENTLE_CENTER_STRENGTH));
            simulation.force("clusterY", d3.forceY(height / 2).strength(GENTLE_CENTER_STRENGTH));
        }

        simulation.force("center", d3.forceCenter(width / 2, height / 2));

        // simulation.alphaDecay(0.02); // Restaurar enfriamiento lento al redimensionar
        // simulation.alpha(0.5).restart();
        if (window.ticked) window.ticked();
        if (window.updateLOD) window.updateLOD();

    }

    function formatKey(key) {
        return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    function createHtmlFromObject(obj) {
        if (!obj) return 'No metadata available';
        let html = '<ul>';
        for (const key in obj) {
            const value = obj[key];
            const formattedKey = formatKey(key);

            if (typeof value === 'object' && value !== null) { // Handles both objects and arrays
                html += `<li>`;
                html += `<details>`; // Ensure it starts collapsed
                html += `<summary><strong>${formattedKey}</strong></summary>`;

                if (Array.isArray(value)) {
                    html += '<ul style="list-style-type: none; padding-left: 10px;">'; // Remove bullet points for the outer list
                    value.forEach((item, index) => {
                        if (typeof item === 'object' && item !== null) {
                            // For objects inside an array, create a collapsible section for each
                            html += `<li>`;
                            html += `<details>`;
                            // Use a generic name or a specific property like 'date' if it always exists
                            const summaryText = item.date ? `Run on ${item.date}` : `Item ${index + 1}`;
                            html += `<summary><strong>${summaryText}</strong></summary>`;
                            html += createHtmlFromObject(item);
                            html += `</details></li>`;
                        } else {
                            // For simple values in an array
                            html += `<li>${item}</li>`;
                        }
                    });
                    html += '</ul>';
                } else {
                    html += createHtmlFromObject(value); // Recursive call for nested objects
                }
                html += `</details></li>`;
            } else {
                html += `<li><strong>${formattedKey}:</strong> ${value}</li>`;
            }
        }
        html += '</ul>';
        return html;
    }

    function displayAllMetadata(allMetadata) {

        const container = document.getElementById('metadata-container');
        let html = '';

        // Iterate directly over the keys of the collected metadata object
        for (const graphName in allMetadata) {
            if (allMetadata.hasOwnProperty(graphName)) {
                html += `<div class="metadata-section">`;
                html += `<h3>${graphName}</h3>`;
                html += createHtmlFromObject(allMetadata[graphName]);
                html += `</div>`;
            }
        }
        container.innerHTML = html;

    }

    // --- PNG Metadata Helpers (DPI Injection) ---
    const crcTable = [];
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) {
            if (c & 1) c = 0xedb88320 ^ (c >>> 1);
            else c = c >>> 1;
        }
        crcTable[n] = c;
    }

    function calcCrc(buf) {
        let c = 0xffffffff;
        for (let i = 0; i < buf.length; i++) {
            c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
        }
        return c ^ 0xffffffff;
    }

    function injectDpi(buffer, dpi) {
        const ppm = Math.round(dpi / 0.0254); // pixels per meter
        const physChunk = new Uint8Array(21);
        const view = new DataView(physChunk.buffer);

        view.setUint32(0, 9); // Length
        view.setUint32(4, 0x70485973); // Type: pHYs
        view.setUint32(8, ppm); // X pixels per unit
        view.setUint32(12, ppm); // Y pixels per unit
        view.setUint8(16, 1); // Unit specifier: meter

        // Calculate CRC for the chunk data (Type + Data)
        const crc = calcCrc(physChunk.slice(4, 17));
        view.setUint32(17, crc);

        // Insert pHYs chunk after IHDR (which is always 33 bytes long in standard PNGs)
        const newBuffer = new Uint8Array(buffer.byteLength + 21);
        newBuffer.set(new Uint8Array(buffer.slice(0, 33)), 0);
        newBuffer.set(physChunk, 33);
        newBuffer.set(new Uint8Array(buffer.slice(33)), 54);

        return newBuffer.buffer;
    }

    // Function to initialize tabs and load the first graph
    async function initialize() {
        // Configure the manual load button (for offline mode)
        const fileInput = document.getElementById('file-input');
        const uploadBtn = document.getElementById('upload-btn');
        const landingPage = document.getElementById('landing-page');

        // Show Landing Page by default
        landingPage.style.display = 'flex';
        document.getElementById('tabs').style.display = 'none';
        document.getElementById('graph-container').style.display = 'none';

        uploadBtn.onclick = () => fileInput.click();
        fileInput.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                loadGraph(file, null);
            }
        };
    }

    // --- Panel Toggle Logic ---
    const panelToggle = document.getElementById('panel-toggle');
    const bottomPanel = document.getElementById('bottom-panels');
    let isResizingPanel = false;

    panelToggle.addEventListener('mousedown', function (e) {
        isResizingPanel = true;
        e.preventDefault(); // Prevent text selection
        document.body.style.cursor = 'ns-resize';
    });

    window.addEventListener('mousemove', function (e) {
        if (!isResizingPanel) return;
        // Calculate new height: Window Height - Mouse Y Position
        const newHeight = window.innerHeight - e.clientY;
        // Limit height (min 50px, max window height - 100px)
        if (newHeight > 50 && newHeight < window.innerHeight - 100) {
            bottomPanel.style.height = `${newHeight}px`;
        }
    });

    window.addEventListener('mouseup', function () {
        if (isResizingPanel) {
            isResizingPanel = false;
            document.body.style.cursor = 'default';
            handleGlobalResize(); // Adjust graph if needed
        }
    });

    // --- Side Panel Toggle Logic ---
    document.getElementById('side-panel-toggle').addEventListener('click', function () {
        const themesPanel = document.getElementById('themes');
        const icon = this.querySelector('svg');
        const isCollapsed = themesPanel.classList.toggle('collapsed');
        icon.style.transform = isCollapsed ? 'rotate(180deg)' : 'rotate(0deg)';
        // No resize call needed, flexbox handles it.
    });

    // --- Minimap Tab Logic ---
    document.querySelectorAll('.minimap-tab-btn').forEach(button => {
        button.addEventListener('click', function () {
            document.querySelectorAll('.minimap-tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.minimap-tab-pane').forEach(pane => pane.classList.remove('active'));
            this.classList.add('active');
            document.getElementById(this.dataset.tab).classList.add('active');
        });
    });

    // Start the application
    initialize();

    // Listen for window resize events to adjust the graph
    window.addEventListener('resize', handleGlobalResize);
})();
