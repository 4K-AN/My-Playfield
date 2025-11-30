import java.util.*;

public class Tugas2_BFS_DFS {
    private int[][] adj;
    private int V;

    public Tugas2_BFS_DFS(int v) {
        V = v;
        adj = new int[v][v];
    }

    public void addEdge(int i, int j) {
        adj[i][j] = 1;
        adj[j][i] = 1;
    }

    public void bfs(int start) {
        boolean[] visited = new boolean[V];
        Queue<Integer> q = new LinkedList<>();
        visited[start] = true;
        q.add(start);

        System.out.print("BFS: ");
        while (!q.isEmpty()) {
            int curr = q.poll();
            System.out.print(curr + " ");
            for (int i = 0; i < V; i++) {
                if (adj[curr][i] == 1 && !visited[i]) {
                    visited[i] = true;
                    q.add(i);
                }
            }
        }
        System.out.println();
    }

    public void dfs(int start) {
        boolean[] visited = new boolean[V];
        Stack<Integer> s = new Stack<>();
        s.push(start);

        System.out.print("DFS: ");
        while (!s.isEmpty()) {
            int curr = s.pop();
            if (!visited[curr]) {
                visited[curr] = true;
                System.out.print(curr + " ");
                for (int i = V - 1; i >= 0; i--) { // Reverse loop for stack order
                    if (adj[curr][i] == 1 && !visited[i]) {
                        s.push(i);
                    }
                }
            }
        }
        System.out.println();
    }

    public static void main(String[] args) {
        Tugas2_BFS_DFS g = new Tugas2_BFS_DFS(5);
        g.addEdge(0, 1); g.addEdge(0, 2);
        g.addEdge(1, 3); g.addEdge(1, 4);
        g.bfs(0);
        g.dfs(0);
    }
}