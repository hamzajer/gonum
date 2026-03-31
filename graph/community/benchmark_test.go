// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package community

import (
	"bufio"
	"compress/gzip"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding/graph6"
	"gonum.org/v1/gonum/graph/graphs/gen"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/internal/order"
)

// arxivEdgePath is the path to the ogbn-arxiv edge CSV file.
// The original dataset is available from Kaggle: https://www.kaggle.com/datasets/dataup1/ogbn-arxiv?resource=download
// Set via -arxiv flag: go test -run TestLeidenVsLouvainArxiv -arxiv /path/to/edge.csv
// To generate graph6 testdata from CSV: go test -run GenerateArxivGraph6 -arxiv /path/to/edge.csv
var arxivEdgePath = flag.String("arxiv", "", "path to ogbn-arxiv edge.csv (enables arxiv-based tests and benchmarks)")
var arxivG6Path = flag.String("arxivg6", "", "path to graph6 file (gzipped OK) representing the arxiv subgraph")

func TestMain(m *testing.M) {
	flag.Parse()
	os.Exit(m.Run())
}

// loadArxivEdges reads all edges from the ogbn-arxiv edge CSV.
// The CSV is expected to have two integer columns per row (src, dst) with no header.
func loadArxivEdges(t testing.TB, path string) [][2]int64 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("failed to open edge file: %v", err)
	}
	defer f.Close()

	r := csv.NewReader(bufio.NewReader(f))
	var edges [][2]int64
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("failed to read edge record: %v", err)
		}
		if len(record) != 2 {
			continue
		}
		src, err1 := strconv.ParseInt(record[0], 10, 64)
		dst, err2 := strconv.ParseInt(record[1], 10, 64)
		if err1 != nil || err2 != nil {
			continue
		}
		edges = append(edges, [2]int64{src, dst})
	}
	return edges
}

// buildArxivSubgraph takes the first maxEdges edges, collects all unique
// nodes involved, remaps them to contiguous IDs [0, N), and returns an
// undirected graph.
func buildArxivSubgraph(t testing.TB, allEdges [][2]int64, maxEdges int) *simple.UndirectedGraph {
	t.Helper()

	if maxEdges > len(allEdges) {
		maxEdges = len(allEdges)
	}
	subset := allEdges[:maxEdges]

	nodeSet := make(map[int64]bool)
	for _, e := range subset {
		nodeSet[e[0]] = true
		nodeSet[e[1]] = true
	}

	origIDs := make([]int64, 0, len(nodeSet))
	for id := range nodeSet {
		origIDs = append(origIDs, id)
	}
	slices.Sort(origIDs)

	remap := make(map[int64]int64, len(origIDs))
	for i, id := range origIDs {
		remap[id] = int64(i)
	}

	g := simple.NewUndirectedGraph()
	for i := range len(origIDs) {
		g.AddNode(simple.Node(int64(i)))
	}

	for _, e := range subset {
		src := remap[e[0]]
		dst := remap[e[1]]
		if src == dst {
			continue
		}
		if g.HasEdgeBetween(src, dst) {
			continue
		}
		g.SetEdge(simple.Edge{F: simple.Node(src), T: simple.Node(dst)})
	}

	return g
}

// arxivSubgraphOrSkip loads the ogbn-arxiv edge CSV and builds a subgraph
// from the first maxEdges edges. The test or benchmark is skipped if the
// -arxiv flag was not provided.
func arxivSubgraphOrSkip(tb testing.TB, maxEdges int) graph.Undirected {
	tb.Helper()
	// Prefer a precomputed graph6 subgraph from flag or testdata when provided.
	var g6path string
	if *arxivG6Path != "" {
		g6path = *arxivG6Path
	} else {
		// Look for testdata files matching the requested size.
		try1 := fmt.Sprintf("testdata/arxiv_%d.graph6.gz", maxEdges)
		try2 := fmt.Sprintf("testdata/arxiv_%d.graph6", maxEdges)
		if _, err := os.Stat(try1); err == nil {
			g6path = try1
		} else if _, err := os.Stat(try2); err == nil {
			g6path = try2
		}
	}

	if g6path != "" {
		f, err := os.Open(g6path)
		if err != nil {
			tb.Fatalf("failed to open graph6 file: %v", err)
		}
		defer f.Close()

		// Detect gzip by magic bytes.
		header := make([]byte, 2)
		n, _ := f.Read(header)
		_, _ = f.Seek(0, io.SeekStart)
		var r io.Reader = f
		if n == 2 && header[0] == 0x1f && header[1] == 0x8b {
			gz, err := gzip.NewReader(f)
			if err != nil {
				tb.Fatalf("failed to create gzip reader: %v", err)
			}
			defer gz.Close()
			r = gz
		}
		data, err := ioutil.ReadAll(r)
		if err != nil {
			tb.Fatalf("failed to read graph6 data: %v", err)
		}
		s := strings.TrimSpace(string(data))
		return graph6.Graph(s)
	}

	if *arxivEdgePath == "" {
		tb.Skip("ogbn-arxiv dataset not provided; use -arxiv /path/to/edge.csv or provide testdata/arxiv_<N>.graph6(.gz) to enable")
	}
	allEdges := loadArxivEdges(tb, *arxivEdgePath)
	return buildArxivSubgraph(tb, allEdges, maxEdges)
}

// TestGenerateArxivGraph6 generates and writes graph6 testdata files from the CSV.
// Run with: go test -run TestGenerateArxivGraph6 -arxiv /path/to/edge.csv
// This creates testdata/arxiv_5000.graph6.gz and testdata/arxiv_10000.graph6.gz
func TestGenerateArxivGraph6(t *testing.T) {
	if *arxivEdgePath == "" {
		t.Skip("ogbn-arxiv dataset not provided; use -arxiv /path/to/edge.csv")
	}

	allEdges := loadArxivEdges(t, *arxivEdgePath)

	for _, maxE := range []int{5000, 10000} {
		g := buildArxivSubgraph(t, allEdges, maxE)
		encoded := graph6.Encode(g)

		// Write plain graph6 file
		plainPath := fmt.Sprintf("testdata/arxiv_%d.graph6", maxE)
		os.MkdirAll("testdata", 0755)
		if err := os.WriteFile(plainPath, []byte(encoded), 0644); err != nil {
			t.Fatalf("failed to write %s: %v", plainPath, err)
		}
		t.Logf("Wrote %s", plainPath)

		// Write gzipped graph6 file
		gzPath := fmt.Sprintf("testdata/arxiv_%d.graph6.gz", maxE)
		gzFile, err := os.Create(gzPath)
		if err != nil {
			t.Fatalf("failed to create %s: %v", gzPath, err)
		}
		gz := gzip.NewWriter(gzFile)
		if _, err := gz.Write([]byte(encoded)); err != nil {
			gz.Close()
			gzFile.Close()
			t.Fatalf("failed to write gzip data: %v", err)
		}
		gz.Close()
		gzFile.Close()
		t.Logf("Wrote %s", gzPath)
	}
}

// BenchmarkLeidenResolution benchmarks the Leiden algorithm across
// multiple resolutions on the duplication-divergence graph.
func BenchmarkLeidenResolution(b *testing.B) {
	type benchCase struct {
		name string
		g    graph.Undirected
	}
	graphs := []benchCase{
		{name: "dupGraph", g: dupGraph},
	}

	for _, bg := range graphs {
		for _, γ := range []float64{0.5, 1, 2, 5, 10} {
			b.Run(fmt.Sprintf("%s/γ=%.1f", bg.name, γ), func(b *testing.B) {
				src := rand.NewPCG(1, 1)
				for i := 0; i < b.N; i++ {
					Leiden(bg.g, γ, src)
				}
			})
		}
	}
}

// BenchmarkLouvainResolution benchmarks the Louvain algorithm across
// multiple resolutions on the duplication-divergence graph.
func BenchmarkLouvainResolution(b *testing.B) {
	for _, γ := range []float64{0.5, 1, 2, 5, 10} {
		b.Run(fmt.Sprintf("dupGraph/γ=%.1f", γ), func(b *testing.B) {
			src := rand.NewPCG(1, 1)
			for i := 0; i < b.N; i++ {
				Modularize(dupGraph, γ, src)
			}
		})
	}
}

// BenchmarkLeidenArxiv benchmarks the Leiden algorithm on ogbn-arxiv subgraphs.
// Enable with: go test -bench=BenchmarkLeidenArxiv -arxiv /path/to/edge.csv
func BenchmarkLeidenArxiv(b *testing.B) {
	for _, maxE := range []int{5000, 10000} {
		g := arxivSubgraphOrSkip(b, maxE)
		for _, γ := range []float64{1, 2, 5} {
			b.Run(fmt.Sprintf("E=%d/γ=%.1f", maxE, γ), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					src := rand.NewPCG(42, uint64(i))
					Leiden(g, γ, src)
				}
			})
		}
	}
}

// BenchmarkLouvainArxiv benchmarks the Louvain algorithm on ogbn-arxiv subgraphs.
// Enable with: go test -bench=BenchmarkLouvainArxiv -arxiv /path/to/edge.csv
func BenchmarkLouvainArxiv(b *testing.B) {
	for _, maxE := range []int{5000, 10000} {
		g := arxivSubgraphOrSkip(b, maxE)
		for _, γ := range []float64{1, 2, 5} {
			b.Run(fmt.Sprintf("E=%d/γ=%.1f", maxE, γ), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					src := rand.NewPCG(42, uint64(i))
					Modularize(g, γ, src)
				}
			})
		}
	}
}

// TestLeidenVsLouvainArxiv compares Leiden and Louvain on ogbn-arxiv
// subgraphs across multiple resolutions. Leiden is expected to identify
// more communities by splitting weakly connected ones that Louvain merges,
// while maintaining comparable modularity.
//
// Enable with: go test -run TestLeidenVsLouvainArxiv -arxiv /path/to/edge.csv
func TestLeidenVsLouvainArxiv(t *testing.T) {
	for _, maxE := range []int{2000, 5000} {
		t.Run(fmt.Sprintf("E=%d", maxE), func(t *testing.T) {
			g := arxivSubgraphOrSkip(t, maxE)
			t.Logf("Subgraph: %d nodes", len(graph.NodesOf(g.Nodes())))

			for _, γ := range []float64{0.5, 1.0, 1.5} {
				t.Run(fmt.Sprintf("γ=%.1f", γ), func(t *testing.T) {
					const iterations = 3

					bestQLouvain, bestQLeiden := math.Inf(-1), math.Inf(-1)
					var bestCommLouvain, bestCommLeiden int

					for i := 0; i < iterations; i++ {
						srcL := rand.New(rand.NewPCG(uint64(i), 0))
						srcD := rand.New(rand.NewPCG(uint64(i), 0))

						rLouvain := Modularize(g, γ, srcL)
						rLeiden := Leiden(g, γ, srcD)

						qL := Q(rLouvain, nil, γ)
						qD := Q(rLeiden, nil, γ)

						if qL > bestQLouvain {
							bestQLouvain = qL
							bestCommLouvain = len(rLouvain.Communities())
						}
						if qD > bestQLeiden {
							bestQLeiden = qD
							bestCommLeiden = len(rLeiden.Communities())
						}
					}

					t.Logf("Louvain: Q=%.6f (%d communities)", bestQLouvain, bestCommLouvain)
					t.Logf("Leiden:  Q=%.6f (%d communities)", bestQLeiden, bestCommLeiden)
					t.Logf("ΔQ=%+.6f  ΔC=%+d",
						bestQLeiden-bestQLouvain, bestCommLeiden-bestCommLouvain)

					// Leiden should not produce drastically worse modularity.
					if bestQLeiden < bestQLouvain-0.05 {
						t.Errorf("Leiden Q (%.6f) significantly worse than Louvain Q (%.6f)",
							bestQLeiden, bestQLouvain)
					}
				})
			}
		})
	}
}

// TestLeidenResolutionEffect verifies that increasing the resolution
// parameter γ produces more (or equal) communities. This holds for
// both standard graphs and the ogbn-arxiv citation network.
//
// The arxiv case is enabled with: go test -run TestLeidenResolutionEffect -arxiv /path/to/edge.csv
func TestLeidenResolutionEffect(t *testing.T) {
	type graphCase struct {
		name    string
		g       graph.Undirected
		skipMsg string // non-empty means skip this case
	}

	cases := []graphCase{
		{name: "dupGraph", g: dupGraph},
	}

	// Include ogbn-arxiv only when the flag is provided.
	if *arxivEdgePath != "" {
		allEdges := loadArxivEdges(t, *arxivEdgePath)
		g := buildArxivSubgraph(t, allEdges, 5000)
		cases = append(cases, graphCase{name: "arxiv_5k", g: g})
	}

	resolutions := []float64{0.5, 1.0, 2.0, 5.0, 10.0}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			prevComms := 0

			for _, γ := range resolutions {
				src := rand.New(rand.NewPCG(1, 1))
				r := Leiden(tc.g, γ, src)
				nComms := len(r.Communities())

				t.Logf("γ=%5.1f → %d communities", γ, nComms)

				if prevComms > 0 && nComms < prevComms {
					t.Errorf("community count decreased from %d (lower γ) to %d at γ=%.1f",
						prevComms, nComms, γ)
				}
				prevComms = nComms
			}
		})
	}
}

// TestLeidenConnectedness verifies that Leiden communities are internally
// connected — the key guarantee over Louvain. Each community of size > 1
// is checked via BFS: all members must be reachable from the first node
// through edges internal to the community.
//
// The arxiv case is enabled with: go test -run TestLeidenConnectedness -arxiv /path/to/edge.csv
func TestLeidenConnectedness(t *testing.T) {
	type graphCase struct {
		name string
		g    graph.Undirected
	}

	cases := []graphCase{
		{name: "dupGraph", g: dupGraph},
	}

	if *arxivEdgePath != "" {
		allEdges := loadArxivEdges(t, *arxivEdgePath)
		g := buildArxivSubgraph(t, allEdges, 5000)
		cases = append(cases, graphCase{name: "arxiv_5k", g: g})
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, γ := range []float64{0.5, 1.0, 2.0, 5.0} {
				t.Run(fmt.Sprintf("γ=%.1f", γ), func(t *testing.T) {
					src := rand.New(rand.NewPCG(1, 1))
					r := Leiden(tc.g, γ, src)

					for i, comm := range r.Communities() {
						if len(comm) <= 1 {
							continue
						}
						if !isCommunityConnected(tc.g, comm) {
							order.ByID(comm)
							t.Errorf("community %d (size %d) is not connected", i, len(comm))
						}
					}
				})
			}
		})
	}
}

// TestLeidenVsLouvainDisconnected demonstrates Louvain's known weakness:
// it can produce communities that are not internally connected.
//
// This uses a 100-node duplication-divergence graph (seed=19) at γ=1.5
// where Louvain merges nodes {8,51,64,70,80,92} into one community,
// but nodes {8,70,80} are not reachable from {51,64,92} via edges
// internal to that community. Leiden, by design, guarantees connected
// communities and also achieves better or comparable modularity.
func TestLeidenVsLouvainDisconnected(t *testing.T) {
	g := simple.NewUndirectedGraph()
	err := gen.Duplication(g, 100, 0.8, 0.1, 0.5, rand.New(rand.NewPCG(19, 0)))
	if err != nil {
		t.Fatalf("failed to generate graph: %v", err)
	}

	const γ = 1.5

	srcLouvain := rand.New(rand.NewPCG(4, 19))
	rLouvain := Modularize(g, γ, srcLouvain)
	louvainComms := rLouvain.Communities()
	qLouvain := Q(rLouvain, nil, γ)

	srcLeiden := rand.New(rand.NewPCG(4, 19))
	rLeiden := Leiden(g, γ, srcLeiden)
	leidenComms := rLeiden.Communities()
	qLeiden := Q(rLeiden, nil, γ)

	t.Logf("Louvain: %d communities, Q=%.6f", len(louvainComms), qLouvain)
	t.Logf("Leiden:  %d communities, Q=%.6f", len(leidenComms), qLeiden)

	// Count and report disconnected communities produced by Louvain.
	louvainDisconnected := 0
	for _, comm := range louvainComms {
		if len(comm) <= 1 {
			continue
		}
		if !isCommunityConnected(g, comm) {
			louvainDisconnected++
			order.ByID(comm)
			ids := make([]int64, len(comm))
			for j, n := range comm {
				ids[j] = n.ID()
			}
			t.Logf("Louvain disconnected community (size %d): %v", len(comm), ids)
		}
	}

	if louvainDisconnected == 0 {
		t.Error("expected at least one disconnected Louvain community on this graph")
	}

	// All Leiden communities must be connected.
	for i, comm := range leidenComms {
		if len(comm) <= 1 {
			continue
		}
		if !isCommunityConnected(g, comm) {
			order.ByID(comm)
			t.Errorf("Leiden community %d (size %d) is not connected", i, len(comm))
		}
	}

	// Leiden should achieve at least comparable modularity.
	if qLeiden < qLouvain-0.05 {
		t.Errorf("Leiden Q (%.6f) significantly worse than Louvain Q (%.6f)", qLeiden, qLouvain)
	}
}

// isCommunityConnected reports whether all nodes in comm form a connected
// subgraph within g. It runs a BFS from the first node, restricted to
// edges whose both endpoints are in the community.
func isCommunityConnected(g graph.Undirected, comm []graph.Node) bool {
	if len(comm) <= 1 {
		return true
	}

	members := make(map[int64]bool, len(comm))
	for _, n := range comm {
		members[n.ID()] = true
	}

	visited := make(map[int64]bool, len(comm))
	queue := []int64{comm[0].ID()}
	visited[comm[0].ID()] = true

	for len(queue) > 0 {
		uid := queue[0]
		queue = queue[1:]
		to := g.From(uid)
		for to.Next() {
			vid := to.Node().ID()
			if members[vid] && !visited[vid] {
				visited[vid] = true
				queue = append(queue, vid)
			}
		}
	}

	return len(visited) == len(comm)
}
