
class NodeCSLL {

    Object data;
    NodeCSLL setelah;
}

public class CircularSingleLinkedListNoPAkhir {

    private NodeCSLL pAwal;

    private int jumlah;

    public CircularSingleLinkedListNoPAkhir() {
        pAwal = null;
        jumlah = -1;
    }

    private NodeCSLL getNodeAkhir() {
        if (pAwal == null) {
            
            return null;
        }
        NodeCSLL pTemp = pAwal;
        while (pTemp.setelah != pAwal) {
            pTemp = pTemp.setelah;
        }
        return pTemp;
    }

    public void SisipDataDiAwal(Object data) {
        NodeCSLL pBaru = new NodeCSLL();
        pBaru.data = data;
        pBaru.setelah = pBaru;

        if (pAwal == null) {
            pAwal = pBaru;
            jumlah = 0;
        } else {
            NodeCSLL pAkhir = getNodeAkhir();
            pBaru.setelah = pAwal;
            pAkhir.setelah = pBaru;
            pAwal = pBaru;
            jumlah++;
        }
    }

    public void SisipDataDiAkhir(Object data) {
        NodeCSLL pBaru = new NodeCSLL();
        pBaru.data = data;
        pBaru.setelah = pBaru;

        if (pAwal == null) {
            pAwal = pBaru;
            jumlah = 0;
        } else {
            NodeCSLL pAkhir = getNodeAkhir();
            pBaru.setelah = pAwal;
            pAkhir.setelah = pBaru;
            jumlah++;
        }
    }

    public void hapusData(Object dtHapus) {
        if (pAwal != null) {
            NodeCSLL pSbl, pKini, pHapus;
            pSbl = null;
            pKini = pAwal;
            boolean ketemu = false;
            int i = 0;

            while (!ketemu && (i <= jumlah)) {
                if (pKini.data.equals(dtHapus)) {
                    ketemu = true;
                } else {
                    pSbl = pKini;
                    pKini = pKini.setelah;
                }
                i++;
            }

            if (ketemu) {
                NodeCSLL pAkhir = getNodeAkhir();

                if (pAwal.setelah == pAwal) {

                    pHapus = pAwal;
                    pAwal = null;
                    jumlah = -1;
                    pHapus = null;
                } else if (pSbl == null) {

                    pHapus = pAwal;
                    pAwal = pKini.setelah;
                    pAkhir.setelah = pAwal;
                    pHapus = null;
                    jumlah--;
                } else {

                    pSbl.setelah = pKini.setelah;
                    pHapus = pKini;
                    pHapus = null;
                    jumlah--;
                }
            }
        }
    }

    public void hapusSatuDataDiAwal() {
        if (pAwal != null) {
            NodeCSLL pHapus;

            if (pAwal.setelah == pAwal) {

                pHapus = pAwal;
                pAwal = null;
                jumlah = -1;
                pHapus = null;
            } else {
                NodeCSLL pAkhir = getNodeAkhir();
                pHapus = pAwal;
                pAwal = pAwal.setelah;
                pAkhir.setelah = pAwal;
                pHapus = null;
                jumlah--;
            }
        }
    }

    public void hapusSatuDataDiAkhir() {
        if (pAwal != null) {
            NodeCSLL pHapus, pBantu;

            if (pAwal.setelah == pAwal) {

                pHapus = pAwal;
                pAwal = null;
                jumlah = -1;
                pHapus = null;
            } else {
                pBantu = pAwal;
                NodeCSLL pAkhir = getNodeAkhir();

                while (pBantu.setelah != pAkhir) {
                    pBantu = pBantu.setelah;
                }
                pHapus = pAkhir;
                pBantu.setelah = pAwal;
                pHapus = null;
                jumlah--;
            }
        }
    }


    public void cetak(String Komentar) {
        System.out.println(Komentar);
        if (pAwal != null) {
            NodeCSLL pCetak = pAwal;
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
        CircularSingleLinkedListNoPAkhir csll = new CircularSingleLinkedListNoPAkhir();

        System.out.println("=== PENGUJIAN CSLL TANPA POINTER pAKHIR ===\n");

        csll.SisipDataDiAwal(new Integer(50));
        csll.SisipDataDiAwal(new Integer(60));
        csll.SisipDataDiAwal(new Integer(70));
        csll.cetak("Data awal");

        csll.SisipDataDiAkhir(new Integer(100));
        csll.cetak("Setelah sisip 100 di akhir");

        csll.hapusData(60);
        csll.cetak("Setelah hapus 60");

        csll.hapusSatuDataDiAwal();
        csll.cetak("Setelah hapus data di awal");

        csll.hapusSatuDataDiAkhir();
        csll.cetak("Setelah hapus data di akhir");
    }
}
