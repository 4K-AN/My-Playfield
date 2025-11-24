package TUGAS.ASD;
import java.util.*;

// Class Edge sama seperti di atas
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

public class PetaMalangBagian2 {
    private Map<String, List<Edge>> adjList;

    public PetaMalangBagian2() {
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

    // c. Fungsi pergi(nodeA, nodeB)
    public void pergi(String start, String end) {
        if (!adjList.containsKey(start) || !adjList.containsKey(end)) {
            System.out.println("jalur tidak dapat dijangkau");
            return;
        }

        System.out.println("---Hasil---");

        // Jalankan BFS
        executeAlgo(start, end, "BFS");
        
        // Jalankan DFS
        executeAlgo(start, end, "DFS");
    }

    private void executeAlgo(String start, String end, String type) {
        Map<String, String> parentMap = new HashMap<>();
        Set<String> visited = new HashSet<>();
        boolean found = false;

        if (type.equals("BFS")) {
            Queue<String> queue = new LinkedList<>();
            queue.add(start);
            visited.add(start);
            parentMap.put(start, null);

            while (!queue.isEmpty()) {
                String curr = queue.poll();
                if (curr.equals(end)) { found = true; break; }

                for (Edge edge : adjList.get(curr)) {
                    if (!visited.contains(edge.destination)) {
                        visited.add(edge.destination);
                        parentMap.put(edge.destination, curr);
                        queue.add(edge.destination);
                    }
                }
            }
        } else { // DFS
            Stack<String> stack = new Stack<>();
            stack.push(start);
            parentMap.put(start, null);

            while (!stack.isEmpty()) {
                String curr = stack.pop();
                if (curr.equals(end)) { found = true; break; }

                if (!visited.contains(curr)) {
                    visited.add(curr);
                    for (Edge edge : adjList.get(curr)) {
                        if (!visited.contains(edge.destination)) {
                            if(!parentMap.containsKey(edge.destination)) parentMap.put(edge.destination, curr);
                            stack.push(edge.destination);
                        }
                    }
                }
            }
        }

        if (found) {
            printResult(end, parentMap, type);
        } else {
            System.out.println("jalur tidak dapat dijangkau dengan " + type);
        }
    }

    private void printResult(String end, Map<String, String> parentMap, String algo) {
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

    public static void main(String[] args) {
        PetaMalangBagian2 peta = new PetaMalangBagian2();
        Scanner sc = new Scanner(System.in);

        // Setup Data
        peta.tambahJalur("Stasiun Kota Baru", "Alun-Alun Merdeka", 2, 10);
        peta.tambahJalur("Alun-Alun Merdeka", "Museum Brawijaya", 3, 10);
        peta.tambahJalur("Museum Brawijaya", "Matos", 3, 5);
        peta.tambahJalur("Matos", "Universitas Brawijaya", 2, 5);
        
        // Jalur Alternatif untuk DFS
        peta.tambahJalur("Stasiun Kota Baru", "Sawojajar", 5, 20);
        peta.tambahJalur("Sawojajar", "Matos", 10, 25);

        System.out.print("Masukkan titik asal : ");
        String asal = sc.nextLine();
        System.out.print("Masukkan titik tujuan: ");
        String tujuan = sc.nextLine();

        peta.pergi(asal, tujuan);
        
        sc.close();
    }
}