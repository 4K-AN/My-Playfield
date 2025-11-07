
class NodeCDLL {

    Object data;
    NodeCDLL sebelum;
    NodeCDLL setelah;
}

public class CircularDoubleLinkedList {

    private NodeCDLL pAwal, pAkhir;
    private int jumlah;

    public CircularDoubleLinkedList() {
        pAwal = null;
        pAkhir = null;
        jumlah = -1;
    }

    public void SisipDataDiAwal(Object data) {
        NodeCDLL pBaru = new NodeCDLL();
        pBaru.data = data;
        pBaru.sebelum = pBaru;
        pBaru.setelah = pBaru;

        if (pAwal == null) {
            pAwal = pBaru;
            pAkhir = pBaru;
            jumlah = 0;
        } else {
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
            pAkhir = pBaru;
            jumlah = 0;
        } else {
            pBaru.sebelum = pAkhir;
            pBaru.setelah = pAwal;
            pAkhir.setelah = pBaru;
            pAwal.sebelum = pBaru;
            pAkhir = pBaru;
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
                if (pAwal == pAkhir) {

                    pHapus = pAwal;
                    pAwal = null;
                    pAkhir = null;
                    jumlah = -1;
                    pHapus = null;
                } else if (pKini == pAwal) {

                    pHapus = pAwal;
                    pAwal = pAwal.setelah;
                    pAwal.sebelum = pAkhir;
                    pAkhir.setelah = pAwal;
                    pHapus = null;
                    jumlah--;
                } else if (pKini == pAkhir) {

                    pHapus = pAkhir;
                    pAkhir = pAkhir.sebelum;
                    pAkhir.setelah = pAwal;
                    pAwal.sebelum = pAkhir;
                    pHapus = null;
                    jumlah--;
                } else {

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
        NodeCDLL pCetak;
        pCetak = pAwal;
        int i = -1;

        while ((i < jumlah)) {
            System.out.print(pCetak.data + "->");
            pCetak = pCetak.setelah;
            i++;
        }
        System.out.println();
    }

    public static void main(String[] args) {
        CircularDoubleLinkedList cdll = new CircularDoubleLinkedList();

        cdll.SisipDataDiAwal(new Integer(50));
        cdll.SisipDataDiAwal(new Integer(60));
        cdll.SisipDataDiAwal(new Integer(70));
        cdll.SisipDataDiAwal(new Integer(8));
        cdll.SisipDataDiAwal(new Integer(9));
        cdll.SisipDataDiAwal(new Integer(90));
        cdll.SisipDataDiAwal(new Integer(19));
        cdll.cetak("cdll Asal");

        System.out.println("\n=== PENGUJIAN PROSEDUR BARU ===");

        cdll.SisipDataDiAkhir(new Integer(100));
        cdll.cetak("cdll stl sisip 100 di akhir");
        cdll.SisipDataDiAkhir(new Integer(200));
        cdll.cetak("cdll stl sisip 200 di akhir");

        cdll.hapusData(19);
        cdll.cetak("cdll stl hapus 19");
        cdll.hapusData(100);
        cdll.cetak("cdll stl hapus 100");
        cdll.hapusData(9);
        cdll.cetak("cdll stl hapus 9");
    }
}
