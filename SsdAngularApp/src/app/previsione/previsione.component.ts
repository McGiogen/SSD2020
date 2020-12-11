import { Component } from '@angular/core';
import { finalize, map } from 'rxjs/operators';
import { Breakpoints, BreakpointObserver } from '@angular/cdk/layout';
import { FormControl, FormGroup } from '@angular/forms';
import { ForecastService } from '../services/proxies/forecast.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-previsione',
  templateUrl: './previsione.component.html',
  styleUrls: ['./previsione.component.css']
})
export class PrevisioneComponent {
  indexes = [{ id: 0, name: 'id'}, { id: 1, name: 'Data'}, { id: 2, name: 'SP_500'}, { id: 3, name: 'FTSE_MIB'}, { id: 4, name: 'GOLD_SPOT'}, { id: 5, name: 'MSCI_EM'}, { id: 6, name: 'MSCI_EURO'}, { id: 7, name: 'All_Bonds'}, { id: 8, name: 'US_Treasury'}]
  types = ['mlp', 'lstm', 'sarima', 'sarimax'];

  form = new FormGroup({
    indiceId: new FormControl(2),
    type: new FormControl('mlp'),
  });

  /** Based on the screen size, switch from standard to one column per row */
  cards = this.breakpointObserver.observe(Breakpoints.Handset).pipe(
    map(({ matches }) => {
      if (matches) {
        return [
          { title: 'Card 1', cols: 2, rows: 1 },
          { title: 'Card 2', cols: 2, rows: 1 },
          { title: 'Card 3', cols: 2, rows: 1 },
          { title: 'Card 4', cols: 2, rows: 1 }
        ];
      }

      return [
        { title: 'Card 1', cols: 2, rows: 1 },
        { title: 'Card 2', cols: 1, rows: 1 },
        { title: 'Card 3', cols: 1, rows: 2 },
        { title: 'Card 4', cols: 1, rows: 1 }
      ];
    })
  );

  result$: Observable<any>;
  loading: boolean = false;

  constructor(
    private breakpointObserver: BreakpointObserver,
    private forecastService: ForecastService,
  ) {}

  previsione() {
    const { indiceId, ...params } = this.form.value;
    this.loading = true;
    this.result$ = this.forecastService.get(Number(indiceId || 2), params)
      .pipe(
        finalize(() => this.loading = false)
      )
  }
}
