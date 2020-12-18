import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

const URL = 'https://localhost:5001/api/Forecast';

@Injectable({
  providedIn: 'root'
})
export class ForecastService {

  constructor(
    private http: HttpClient,
  ) { }

  list(): Observable<any> {
    return this.http.get<any>(URL);
  }

  get(id: number, queryParams): Observable<{ text: string, img: string[] }> {
    return this.http.get<{ text: string, img: string[] }>(`${URL}/${id}`, { params: queryParams });
  }
}
