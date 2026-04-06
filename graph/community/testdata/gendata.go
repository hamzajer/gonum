// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/graph/encoding/graph6"
	"gonum.org/v1/gonum/graph/simple"
)

func main() {
	max := flag.Int("max", 0, "maximum edges")
	flag.Parse()

	g := simple.NewUndirectedGraph()
	sc := bufio.NewScanner(os.Stdin)
	for i := 0; i < *max && sc.Scan(); i++ {
		f, t, ok := strings.Cut(sc.Text(), ",")
		if !ok {
			log.Fatal("no comma")
		}
		fid, err := strconv.Atoi(f)
		if err != nil {
			log.Fatalf("from: %v", err)
		}
		tid, err := strconv.Atoi(t)
		if err != nil {
			log.Fatalf("from: %v", err)
		}
		g.SetEdge(simple.Edge{F: simple.Node(fid), T: simple.Node(tid)})
	}
	fmt.Println(graph6.Encode(g))
}
