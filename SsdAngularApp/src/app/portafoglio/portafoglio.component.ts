import { Component } from '@angular/core';
import { finalize, map } from 'rxjs/operators';
import { PortafoglioService } from '../services/proxies/portafoglio.service'
import { FormControl, FormGroup } from '@angular/forms';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-portafoglio',
  templateUrl: './portafoglio.component.html',
  styleUrls: ['./portafoglio.component.css']
})
export class PortafoglioComponent {
  indexes = ['SP_500', 'FTSE_MIB', 'GOLD_SPOT', 'MSCI_EM', 'MSCI_EURO', 'All_Bonds', 'US_Treasury']
  types = ['mlp', 'lstm', 'sarima', 'sarimax'];
  risks = ['var', 'mape'];

  form = new FormGroup({
    type: new FormControl(this.types[2]),
    risk: new FormControl(this.risks[0]),
  });

  result$: Observable<any>;
  loading: boolean = false;

  constructor(
    private portafoglioService: PortafoglioService,
  ) {}

  calcolaPortafoglio() {
    this.loading = true;
    this.result$ = this.portafoglioService.list(this.form.value)
      .pipe(
        finalize(() => this.loading = false)
      )
  }

}
