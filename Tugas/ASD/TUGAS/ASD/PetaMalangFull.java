package TUGAS.ASD;
import java.util.*;

class Edge {
    String destination;
    int jarak; 
    int waktu; 

    public Edge(String destination, int jarak, int waktu) {
        this.destination = destination;
        this.jarak = jarak;
        this.waktu = waktu;
    }
}

class NodeJarak implements Comparable<NodeJarak> {
    String id;
    int jarak;

    public NodeJarak(String id, int jarak) {
        this.id = id;
        this.jarak = jarak;
    }

    @Override
    public int compareTo(NodeJarak other) {
        return Integer.compare(this.jarak, other.jarak); // Ascending (Kecil ke Besar)
    }
}

public class PetaMalangFull {
    private Map<String, List<Edge>> adjList;

    public PetaMalangFull() {
        this.adjList = new HashMap<>();
    }

    public void tambahLokasi(String node) {
        adjList.putIfAbsent(node, new ArrayList<>());
    }

    public void tambahJalur(String nodeA, String nodeB, int jarak, int waktu) {
        tambahLokasi(nodeA);
        tambahLokasi(nodeB);

        adjList.get(nodeA).add(new Edge(nodeB, jarak, waktu));
        adjList.get(nodeB).add(new Edge(nodeA, jarak, waktu));
    }

    public boolean bisaPergi(String start, String end) {
        if (!adjList.containsKey(start) || !adjList.containsKey(end)) return false;

        Queue<String> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();

        queue.add(start);
        visited.add(start);

        while (!queue.isEmpty()) {
            String current = queue.poll();
            if (current.equals(end)) return true;

            for (Edge edge : adjList.get(current)) {
                if (!visited.contains(edge.destination)) {
                    visited.add(edge.destination);
                    queue.add(edge.destination);
                }
            }
        }
        return false;
    }

    public void pergi(String start, String end) {
        if (!adjList.containsKey(start) || !adjList.containsKey(end)) {
            System.out.println("Lokasi tidak valid.");
            return;
        }

        executeTraversal(start, end, "BFS");

        executeTraversal(start, end, "DFS");
    }

    private void executeTraversal(String start, String end, String algo) {
        Map<String, String> parentMap = new HashMap<>();
        Set<String> visited = new HashSet<>();
        boolean found = false;

        if (algo.equals("BFS")) {
            Queue<String> queue = new LinkedList<>();
            queue.add(start);
            visited.add(start);
            parentMap.put(start, null);

            while (!queue.isEmpty()) {
                String current = queue.poll();
                if (current.equals(end)) { found = true; break; }

                for (Edge edge : adjList.get(current)) {
                    if (!visited.contains(edge.destination)) {
                        visited.add(edge.destination);
                        parentMap.put(edge.destination, current);
                        queue.add(edge.destination);
                    }
                }
            }
        } else { 
            Stack<String> stack = new Stack<>();
            stack.push(start);
            parentMap.put(start, null);

            while (!stack.isEmpty()) {
                String current = stack.pop();
                if (current.equals(end)) { found = true; break; }

                if (!visited.contains(current)) {
                    visited.add(current);
                    for (Edge edge : adjList.get(current)) {
                        if (!visited.contains(edge.destination)) {
                            if (!parentMap.containsKey(edge.destination)) {
                                parentMap.put(edge.destination, current);
                            }
                            stack.push(edge.destination);
                        }
                    }
                }
            }
        }

        if (found) {
            printPath(end, parentMap, algo);
        } else {
            System.out.println("Jalur tidak dapat dijangkau dengan " + algo);
        }
    }

    private void printPath(String end, Map<String, String> parentMap, String algo) {
        LinkedList<String> path = new LinkedList<>();
        String curr = end;
        while (curr != null) {
            path.addFirst(curr);
            curr = parentMap.get(curr);
        }

        int totalJarak = 0;
        int totalWaktu = 0;

        for (int i = 0; i < path.size() - 1; i++) {
            String u = path.get(i);
            String v = path.get(i + 1);
            for (Edge e : adjList.get(u)) {
                if (e.destination.equals(v)) {
                    totalJarak += e.jarak;
                    totalWaktu += e.waktu;
                    break;
                }
            }
        }

        System.out.println("Menggunakan " + algo + ", total jarak: " + totalJarak + "km, total waktu: " + totalWaktu + "menit");
        System.out.println("Jalur " + String.join(", ", path));
    }

    public void jalurTerpendek(String start, String end) {
        if (!adjList.containsKey(start) || !adjList.containsKey(end)) return;

        Map<String, Integer> distances = new HashMap<>();
        Map<String, String> parentMap = new HashMap<>();
        PriorityQueue<NodeJarak> pq = new PriorityQueue<>();
        Set<String> visited = new HashSet<>();

        for (String node : adjList.keySet()) {
            distances.put(node, Integer.MAX_VALUE);
        }
        distances.put(start, 0);
        pq.add(new NodeJarak(start, 0));

        while (!pq.isEmpty()) {
            NodeJarak current = pq.poll();
            String u = current.id;

            if (u.equals(end)) break; 
            if (visited.contains(u)) continue;
            visited.add(u);

            if (adjList.containsKey(u)) {
                for (Edge edge : adjList.get(u)) {
                    String v = edge.destination;
                    int newDist = distances.get(u) + edge.jarak;

                    if (newDist < distances.get(v)) {
                        distances.put(v, newDist);
                        parentMap.put(v, u);
                        pq.add(new NodeJarak(v, newDist));
                    }
                }
            }
        }

        if (distances.get(end) == Integer.MAX_VALUE) {
            System.out.println("Jalur Dijkstra tidak ditemukan.");
        } else {
            System.out.println("Menggunakan Dijkstra (Jarak Terpendek), total jarak: " + distances.get(end) + "km");
            LinkedList<String> path = new LinkedList<>();
            String curr = end;
            while (curr != null) {
                path.addFirst(curr);
                curr = parentMap.get(curr);
            }
            System.out.println("Jalur " + String.join(", ", path));
        }
    }

    public static void main(String[] args) {
        PetaMalangFull peta = new PetaMalangFull();
        Scanner sc = new Scanner(System.in);

        peta.tambahJalur("Stasiun Kota Baru", "Alun-Alun Merdeka", 2, 10);
        peta.tambahJalur("Stasiun Kota Baru", "Pasar Besar", 3, 15);
        peta.tambahJalur("Alun-Alun Merdeka", "Museum Brawijaya", 3, 10);
        peta.tambahJalur("Museum Brawijaya", "Matos", 3, 5);
        peta.tambahJalur("Matos", "Universitas Brawijaya", 2, 5);
        
        peta.tambahJalur("Stasiun Kota Baru", "Sawojajar", 5, 20);
        peta.tambahJalur("Sawojajar", "Universitas Brawijaya", 15, 40);

        System.out.print("Masukkan titik asal : ");
        String asal = sc.nextLine();
        System.out.print("Masukkan titik tujuan: ");
        String tujuan = sc.nextLine();

        System.out.println("\n---Hasil----");
        
        boolean bisa = peta.bisaPergi(asal, tujuan);
        System.out.println("Dari " + asal + " menuju " + tujuan + ": (BFS) " + (bisa ? "Dapat dijangkau" : "Tidak dapat dijangkau"));

        if (bisa) {
            peta.pergi(asal, tujuan);
            
            System.out.println();
            peta.jalurTerpendek(asal, tujuan);
        }

        sc.close();
    }
}