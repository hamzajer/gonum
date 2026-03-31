// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package community

import (
	"bytes"
	"compress/gzip"
	"embed"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding/graph6"
	"gonum.org/v1/gonum/graph/graphs/gen"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/internal/order"
)

// the original data sets are from https://www.kaggle.com/datasets/dataup1/ogbn-arxiv
//
//go:embed testdata/*.graph6.gz
var testdataGraphs embed.FS

// loadGraph6 loads a graph6-encoded graph from embedded testdata.
// The filename should be relative to testdata (e.g., "arxiv_5000.graph6.gz").
// Returns nil if the file is not found.
func loadGraph6(filename string) graph.Undirected {
	path := "testdata/" + filename
	data, err := testdataGraphs.ReadFile(path)
	if err != nil {
		return nil
	}

	var r io.Reader = bytes.NewReader(data)

	// Detect gzip by magic bytes.
	if len(data) >= 2 && data[0] == 0x1f && data[1] == 0x8b {
		gz, err := gzip.NewReader(bytes.NewReader(data))
		if err != nil {
			return nil
		}
		defer gz.Close()
		r = gz
	}

	decoded, err := io.ReadAll(r)
	if err != nil {
		return nil
	}

	s := strings.TrimSpace(string(decoded))
	return graph6.Graph(s)
}

// arxivSubgraphOrSkip loads the ogbn-arxiv subgraph from embedded testdata.
// The test or benchmark is skipped if the testdata file is not found.
func arxivSubgraphOrSkip(tb testing.TB, maxEdges int) graph.Undirected {
	tb.Helper()

	filename := fmt.Sprintf("arxiv_%d.graph6.gz", maxEdges)
	g := loadGraph6(filename)
	if g == nil {
		tb.Skipf("testdata/%s not found", filename)
	}
	return g
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
func TestLeidenVsLouvainArxiv(t *testing.T) {
	// 10000 edges is too slow
	//for _, maxE := range []int{5000,10000} {
	for _, maxE := range []int{5000} {
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
func TestLeidenResolutionEffect(t *testing.T) {
	type graphCase struct {
		name    string
		g       graph.Undirected
		skipMsg string // non-empty means skip this case
	}

	cases := []graphCase{
		{name: "dupGraph", g: dupGraph},
	}

	// Try to include ogbn-arxiv if available.
	g := loadGraph6("arxiv_5000.graph6.gz")
	if g != nil {
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
func TestLeidenConnectedness(t *testing.T) {
	type graphCase struct {
		name string
		g    graph.Undirected
	}

	cases := []graphCase{
		{name: "dupGraph", g: dupGraph},
	}

	// Try to include ogbn-arxiv if available.
	g := loadGraph6("arxiv_5000.graph6.gz")
	if g != nil {
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
