import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

const URL = '/api/Portafoglio';

@Injectable({
  providedIn: 'root'
})
export class PortafoglioService {

  constructor(
    private http: HttpClient,
  ) { }

  list(queryParams?: { type: string }): Observable<any> {
    return this.http.get<{ text: string[], img: string[][], portfolio: object }>(URL, { params: queryParams });
  }
}
