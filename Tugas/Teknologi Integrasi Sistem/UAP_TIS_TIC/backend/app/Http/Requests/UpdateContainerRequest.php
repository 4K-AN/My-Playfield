<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Contracts\Validation\Validator;
use Illuminate\Http\Exceptions\HttpResponseException;

class UpdateContainerRequest extends FormRequest
{
    /**
     * Determine if the user is authorized to make this request.
     */
    public function authorize(): bool
    {
        return true;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array<string, \Illuminate\Contracts\Validation\ValidationRule|array<mixed>|string>
     */
    public function rules(): array
    {
        $containerId = $this->route('id');

        $rules = [
            'container_code' => "sometimes|string|unique:containers,container_code,{$containerId}|max:50",
            'type' => 'sometimes|in:limbah_cair,limbah_padat,limbah_gas',
            'capacity' => 'sometimes|numeric|min:1',
            'current_fill_level' => 'sometimes|numeric|min:0',
            'location' => 'sometimes|string|max:255',
            'status' => 'sometimes|in:Active,Maintenance,Archived,Full',
            'last_maintenance_date' => 'nullable|date',
        ];

        // Conditional: jika mengubah status ke Full, fill_level harus ada
        if ($this->input('status') === 'Full') {
            $rules['current_fill_level'] = 'required|numeric|min:0';
        }

        return $rules;
    }

    /**
     * Get custom validation messages.
     */
    public function messages(): array
    {
        return [
            'container_code.unique' => 'Kode kontainer sudah digunakan.',
            'type.in' => 'Tipe limbah harus salah satu dari: limbah_cair, limbah_padat, limbah_gas.',
            'capacity.min' => 'Kapasitas minimal adalah 1.',
            'status.in' => 'Status harus salah satu dari: Active, Maintenance, Archived, Full.',
        ];
    }

    /**
     * Handle a failed validation attempt.
     */
    protected function failedValidation(Validator $validator): void
    {
        throw new HttpResponseException(response()->json([
            'success' => false,
            'message' => 'Validasi gagal',
            'errors' => $validator->errors(),
        ], 422));
    }
}
