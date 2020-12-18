import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Quotazione } from '../../model/quotazione';
import { Observable } from 'rxjs';

const URL = 'https://localhost:5001/api/Quotazione';

@Injectable({
  providedIn: 'root'
})
export class QuotazioneService {

  constructor(
    private http: HttpClient,
  ) { }

  list(): Observable<Quotazione> {
    return this.http.get<Quotazione>(URL);
  }

  get(id: number): Observable<Quotazione> {
    return this.http.get<Quotazione>(`${URL}/${id}`);
  }

  create(data: Omit<Quotazione, 'id'>): Observable<Quotazione> {
    return this.http.post<Quotazione>(URL, data);
  }

  update(data: Quotazione): Observable<Quotazione> {
    return this.http.put<Quotazione>(`${URL}/${data.id}`, data);
  }

  delete(id: number): Observable<Quotazione> {
    return this.http.delete<Quotazione>(`${URL}/${id}`);
  }
}
