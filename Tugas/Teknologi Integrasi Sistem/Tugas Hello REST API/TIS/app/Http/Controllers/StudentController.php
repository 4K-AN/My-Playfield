<?php
// NIM 245150707111012 , NAMA : AKHMAD SYAFIUL ANAM

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class StudentController extends Controller
{
    private $storageFile = 'storage/app/students.json';
    
    private function loadStudents()
    {
        if (!file_exists($this->storageFile)) {
            $defaultData = [
                [
                    "nim" => "245150707111012",
                    "nama" => "Citra Dewi",
                    "mataKuliah" => [
                        ["kode" => "CIE61205", "nama" => "PemWeb", "sks" => 3],
                        ["kode" => "COM60015", "nama" => "MatDis", "sks" => 2]
                    ]
                ],
                [
                    "nim" => "245150707111013",
                    "nama" => "Andy Lau",
                    "mataKuliah" => [
                        ["kode" => "CIE61205", "nama" => "PemWeb", "sks" => 3],
                        ["kode" => "CIE61206", "nama" => "JarKom", "sks" => 3],
                        ["kode" => "CIE61208", "nama" => "BasDat", "sks" => 3],
                    ]
                ]
            ];
            $this->saveStudents($defaultData);
            return $defaultData;
        }
        return json_decode(file_get_contents($this->storageFile), true) ?? [];
    }
    
    private function saveStudents($data)
    {
        $dir = dirname($this->storageFile);
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }
        file_put_contents($this->storageFile, json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
    }

    public function index()
    {
        return response()->json($this->loadStudents());
    }

    public function store(Request $request)
    {
        try {
            $validated = $request->validate([
                'nim' => 'required|digits:15',
                'nama' => 'required|string|min:3|max:50',
                'mataKuliah' => 'required|array|min:1',
                'mataKuliah.*.kode' => 'required|regex:/^[A-Z]{3}[0-9]{5}$/',
                'mataKuliah.*.nama' => 'required|string|max:50',
                'mataKuliah.*.sks' => 'required|numeric|min:1|max:6',
            ]);
        } catch (\Illuminate\Validation\ValidationException $th) {
            return response()->json([
                "message" => "Validation failed",
                "errors" => $th->validator->errors()
            ], 422);
        }

        $students = $this->loadStudents();
        $newStudent = [
            "nim" => $validated['nim'],
            "nama" => $validated['nama'],
            "mataKuliah" => $validated['mataKuliah']
        ];
        
        $found = false;
        foreach ($students as &$std) {
            if ($std['nim'] == $validated['nim']) {
                $std = $newStudent;
                $found = true;
                break;
            }
        }
        
        if (!$found) {
            $students[] = $newStudent;
        }
        
        $this->saveStudents($students);

        return response()->json([
            "message" => "Student created/updated successfully",
            "data" => $newStudent
        ], 201);
    }

    
    public function update(Request $request, $nim)
    {
        try {
            $validated = $request->validate([
                'nama' => 'sometimes|required|string|min:3|max:50',
                'mataKuliah' => 'sometimes|required|array|min:1',
                'mataKuliah.*.kode' => 'sometimes|required|regex:/^[A-Z]{3}[0-9]{5}$/',
                'mataKuliah.*.nama' => 'sometimes|required|string|max:50',
                'mataKuliah.*.sks' => 'sometimes|required|numeric|min:1|max:6',
            ]);
        } catch (\Illuminate\Validation\ValidationException $th) {
            return response()->json([
                "message" => "Validation failed",
                "errors" => $th->validator->errors()
            ], 422);
        }

        $students = $this->loadStudents();
        $found = false;
        
        foreach ($students as &$std) {
            if ($std['nim'] == $nim) {
                if (isset($validated['nama'])) {
                    $std['nama'] = $validated['nama'];
                }
                if (isset($validated['mataKuliah'])) {
                    $std['mataKuliah'] = $validated['mataKuliah'];
                }
                $found = true;
                break;
            }
        }
        
        if (!$found) {
            return response()->json([
                "message" => "Student not found",
                "error" => "NIM {$nim} tidak ditemukan"
            ], 404);
        }
        
        $this->saveStudents($students);

        return response()->json([
            "message" => "Student {$nim} updated successfully",
            "data" => $students
        ]);
    }

    
    public function destroy($nim)
    {
        $students = $this->loadStudents();
        $found = false;
        
        foreach ($students as $key => $std) {
            if ($std['nim'] == $nim) {
                unset($students[$key]);
                $found = true;
                break;
            }
        }
        
        if (!$found) {
            return response()->json([
                "message" => "Student not found",
                "error" => "NIM {$nim} tidak ditemukan"
            ], 404);
        }
        
        $students = array_values($students);
        $this->saveStudents($students);

        return response()->json([
            "message" => "Student {$nim} deleted successfully",
            "data" => $students
        ]);
    }

    public function show($nim)
    {
        $students = $this->loadStudents();
        
        foreach ($students as $student) {
            if ($student['nim'] === $nim) {
                return response()->json([
                    "message" => "Student retrieved successfully",
                    "data" => $student
                ], 200);
            }
        }
        
        return response()->json([
            "message" => "Student not found",
            "error" => "NIM {$nim} tidak ditemukan"
        ], 404);
    }

    public function mataKuliahByStudent($nim)
    {
        $students = $this->loadStudents();
        
        foreach ($students as $student) {
            if ($student['nim'] === $nim) {
                return response()->json([
                    "message" => "Courses retrieved successfully",
                    "student_nim" => $nim,
                    "data" => $student['mataKuliah']
                ], 200);
            }
        }
        
        return response()->json([
            "message" => "Student not found",
            "error" => "NIM {$nim} tidak ditemukan"
        ], 404);
    }

    public function search(Request $request)
    {
        $nim = $request->query('nim');
        $nama = $request->query('nama');
        $kode_mk = $request->query('kode_mk');

        if (!$nim && !$nama && !$kode_mk) {
            return response()->json([
                "error" => "Parameter tidak ditemukan. Harap masukkan nim, nama, atau kode_mk."
            ], 400);
        }

        $students = $this->loadStudents();
        $results = [];

        foreach ($students as $student) {
            $match = false;
            
            if ($nim && $student['nim'] == $nim) {
                $match = true;
            }
            elseif ($nama && stripos($student['nama'], $nama) !== false) {
                $match = true;
            }
            elseif ($kode_mk) {
                foreach ($student['mataKuliah'] as $mk) {
                    if ($mk['kode'] === $kode_mk) {
                        $match = true;
                        break;
                    }
                }
            }

            if ($match) {
                $results[] = $student;
            }
        }

        return response()->json([
            "message" => "Hasil pencarian",
            "data" => $results
        ]);
    }
}