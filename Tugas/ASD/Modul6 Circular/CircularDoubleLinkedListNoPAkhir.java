
class NodeCDLL {

    Object data;
    NodeCDLL sebelum;
    NodeCDLL setelah;
}

public class CircularDoubleLinkedListNoPAkhir {

    private NodeCDLL pAwal;
    private int jumlah;

    public CircularDoubleLinkedListNoPAkhir() {
        pAwal = null;
        jumlah = -1;
    }

    // Method bantu untuk mendapatkan node akhir
    private NodeCDLL getNodeAkhir() {
        if (pAwal == null) {
            return null;
        }
        return pAwal.sebelum;
    }

    public void SisipDataDiAwal(Object data) {
        NodeCDLL pBaru = new NodeCDLL();
        pBaru.data = data;
        pBaru.sebelum = pBaru;
        pBaru.setelah = pBaru;

        if (pAwal == null) {
            pAwal = pBaru;
            jumlah = 0;
        } else {
            NodeCDLL pAkhir = getNodeAkhir();
            pBaru.sebelum = pAkhir;
            pBaru.setelah = pAwal;
            pAwal.sebelum = pBaru;
            pAkhir.setelah = pBaru;
            pAwal = pBaru;
            jumlah++;
        }
    }

    public void SisipDataDiAkhir(Object data) {
        NodeCDLL pBaru = new NodeCDLL();
        pBaru.data = data;
        pBaru.sebelum = pBaru;
        pBaru.setelah = pBaru;

        if (pAwal == null) {
            pAwal = pBaru;
            jumlah = 0;
        } else {
            NodeCDLL pAkhir = getNodeAkhir();
            pBaru.sebelum = pAkhir;
            pBaru.setelah = pAwal;
            pAkhir.setelah = pBaru;
            pAwal.sebelum = pBaru;
            jumlah++;
        }
    }

    public void hapusData(Object dtHapus) {
        if (pAwal != null) {
            NodeCDLL pKini, pHapus;
            pKini = pAwal;
            boolean ketemu = false;
            int i = 0;

            while (!ketemu && (i <= jumlah)) {
                if (pKini.data.equals(dtHapus)) {
                    ketemu = true;
                } else {
                    pKini = pKini.setelah;
                }
                i++;
            }

            if (ketemu) {
                if (pAwal.setelah == pAwal) {
                    // Hanya ada satu node
                    pHapus = pAwal;
                    pAwal = null;
                    jumlah = -1;
                    pHapus = null;
                } else if (pKini == pAwal) {
                    // Hapus node awal
                    NodeCDLL pAkhir = getNodeAkhir();
                    pHapus = pAwal;
                    pAwal = pAwal.setelah;
                    pAwal.sebelum = pAkhir;
                    pAkhir.setelah = pAwal;
                    pHapus = null;
                    jumlah--;
                } else {
                    // Hapus node di tengah atau akhir
                    pHapus = pKini;
                    pKini.sebelum.setelah = pKini.setelah;
                    pKini.setelah.sebelum = pKini.sebelum;
                    pHapus = null;
                    jumlah--;
                }
            }
        }
    }

    public void cetak(String Komentar) {
        System.out.println(Komentar);
        if (pAwal != null) {
            NodeCDLL pCetak = pAwal;
            int i = -1;
            while (i < jumlah) {
                System.out.print(pCetak.data + "->");
                pCetak = pCetak.setelah;
                i++;
            }
            System.out.println();
        } else {
            System.out.println("List Kosong");
        }
    }

    public static void main(String[] args) {
        CircularDoubleLinkedListNoPAkhir cdll = new CircularDoubleLinkedListNoPAkhir();

        System.out.println("=== PENGUJIAN CDLL TANPA POINTER pAKHIR ===\n");

        cdll.SisipDataDiAwal(new Integer(50));
        cdll.SisipDataDiAwal(new Integer(60));
        cdll.SisipDataDiAwal(new Integer(70));
        cdll.cetak("Data awal");

        cdll.SisipDataDiAkhir(new Integer(100));
        cdll.cetak("Setelah sisip 100 di akhir");

        cdll.hapusData(60);
        cdll.cetak("Setelah hapus 60");

        cdll.hapusData(70);
        cdll.cetak("Setelah hapus 70");

        cdll.hapusData(100);
        cdll.cetak("Setelah hapus 100");
    }
}
